import numpy as np
import torch
from torch import nn
from copy import deepcopy

# import IPython
# if "Interactive" in str(IPython.get_ipython()):
#     from tqdm.notebook import tqdm
# else:
#     import tqdm
import re
import psutil
if any(re.search('jupyter-lab-script', x)
       for x in psutil.Process().parent().cmdline()):
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class EWC:
    def calc_ipts(self, model, dataloader, loss_fn, ipt_groups):
        device = next(model.parameters()).device.type
        cur_size = len(dataloader)

        # prev_fisher = {}
        # for key, ipt in ipt_groups.items():
        #     prev_fisher[key] = ipt.clone().detach()
        #     ipt.zero_()

        # trash
        for key, ipt in ipt_groups.items():
            ipt.zero_()

        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            model.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if name not in ipt_groups:
                    continue
                ipt_groups[name] += param.grad.pow(2)

        # for key, ipt in ipt_groups.items():
        #     ipt.div_(cur_size)
        #     ipt.add_(prev_fisher[key])

        # trash
        for key, ipt in ipt_groups.items():
            ipt.div_(cur_size)


class MAS:
    def calc_ipts(self, model, dataloader, ipt_groups, total_size):
        device = next(model.parameters()).device.type
        cur_size = len(dataloader)
        prev_size = total_size - cur_size

        # for key in ipt_groups.keys():
        #     ipt_groups[key].mul_(prev_size)

        # trash
        for key in ipt_groups.keys():
            ipt_groups[key].zero_()

        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            final = torch.norm(pred, p='fro').pow(2)
            model.zero_grad()
            final.backward()
            for name, param in model.named_parameters():
                if name not in ipt_groups:
                    continue
                ipt_groups[name] += param.grad.abs()

        # for ipt in ipt_groups.values():
        #     ipt /= total_size

        # trash
        for ipt in ipt_groups.values():
            ipt /= cur_size


class SCP:
    def __init__(self):
        self.slice_seed = 247
        self.sample_length = 100

    def calc_ipts(self, model, dataloader, ipt_groups, task_id):
        device = next(model.parameters()).device.type
        cur_size = len(dataloader)
        mean_pred = 0
        
        prev_fisher = {}
        for key, value in ipt_groups.items():
            prev_fisher[key] = value.clone().detach()
            value.zero_()

        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            # Compute prediction error
            pred = model(X)
            mean_pred += pred.sum(dim=0)
        mean_pred /= cur_size

        # sample from sphere space
        pbar = tqdm(total=self.sample_length)
        for i in range(self.sample_length):
            torch.manual_seed(self.slice_seed + task_id*self.sample_length + i)
            sample = torch.rand_like(mean_pred.data)
            norm_sample = sample / torch.norm(sample).clamp(min=1e-12)
            response = (mean_pred * norm_sample).sum()
            model.zero_grad()
            response.backward(retain_graph=True)
            for name, param in model.named_parameters():
                if name not in ipt_groups:
                    continue
                ipt_groups[name] += param.grad.pow(2)
            pbar.update(1)
        pbar.close()
        # free memory
        response.backward()
        
        for key, ipt in ipt_groups.items():
            ipt.div_(self.sample_length)
            ipt_groups[key] = 0.5 * (ipt + prev_fisher[key])


class Neuron:
    def calc_ipts(self, model, ipt_groups, dataloader, total_size):
        device = next(model.parameters()).device.type
        cur_size = len(dataloader)
        prev_size = total_size - cur_size

        rec_fms = self.save_neurons(model, dataloader, ipt_groups)

        for value in ipt_groups.values():
            value.mul_(prev_size)

        for name, fms in rec_fms.items():
            neuron_ipt = np.mean(np.concatenate(fms), axis=0)
            neuron_ipt /= (np.sum(neuron_ipt) / neuron_ipt.size)
            in_features = ipt_groups[name + ".weight"].shape[1]
            if "fc" in name:
                w_ipt = torch.from_numpy(np.tile(neuron_ipt, (in_features,1)).T)
                b_ipt = torch.from_numpy(neuron_ipt)
            else:
                w, h = ipt_groups[name + ".weight"].shape[2:]
                channel_ipt = np.linalg.norm(neuron_ipt, axis=(1,2))
                w_ipt = np.expand_dims(channel_ipt, axis=(1,2,3))
                w_ipt = np.tile(w_ipt, (1,in_features,w,h))
                w_ipt = torch.from_numpy(w_ipt)
                b_ipt = torch.from_numpy(channel_ipt)
            ipt_groups[name + ".weight"] += w_ipt.to(device)
            ipt_groups[name + ".bias"] += b_ipt.to(device)

        for value in ipt_groups.values():
            value /= total_size
            value.abs_()

    def save_neurons(self, model, dataloader, ipt_groups):
        device = next(model.parameters()).device.type

        def fhook(module, f_in, f_out):
            nonlocal module_id, amount
            module_name = layers[module_id]
            rec_fms[module_name].append(f_out.clone().detach().cpu().numpy())
            module_id += 1
            if module_id >= amount:
                module_id = 0

        layers = self.register_layers(model, ipt_groups)
        rec_fms = self.init_rec_fms(layers)
        module_id = 0
        amount = len(layers)

        handles = self.add_handles(model, layers, fhook)

        print("Start capturing feature mapping.")
        print("------------------------------------------")
        pbar = tqdm(total=len(dataloader))
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            model(X)
            pbar.update(1)
        pbar.close()
        print("------------------------------------------")

        self.remove_handles(handles)
        return rec_fms

    def register_layers(self, model, ipt_groups):
        layers = list()
        for name, param in model.named_parameters():
            if name not in ipt_groups:
                continue
            if "weight" not in name:
                continue
            layers.append(name.replace(".weight", ""))
        return layers

    def init_rec_fms(self, layers):
        rec_fms = dict()
        for layer in layers:
            rec_fms[layer] = list()

        return rec_fms

    def add_handles(self, model, layers, fhook):
        handles = []
        for name, module in model.named_modules():
            if name in layers:
                handles.append(module.register_forward_hook(fhook))
        return handles

    def remove_handles(self, handles):
        # 删除句柄
        for handle in handles:
            handle.remove()


class SI:
    def __init__(self):
        self.eps = 1e-3

    def update_omegas(self, model, reg_groups):
        omegas = reg_groups["omegas"]
        last_params = reg_groups["last_params"]

        for name, param in model.named_parameters():
            if name not in omegas:
                continue
            cur_diff = last_params[name] - param.data
            omega = param.grad.clone().detach()*cur_diff
            omegas[name] += omega

    def update_last_params(self, model, reg_groups):
        last_params = reg_groups["last_params"]
        for name, param in model.named_parameters():
            last_params[name] = param.clone().detach()

    def calc_ipts(self, model, reg_groups):
        ipt_groups = reg_groups["ipt_groups"]
        prev_params = reg_groups["prev_params"]
        omegas = reg_groups["omegas"]

        for name, param in model.named_parameters():
            if name not in ipt_groups:
                continue
            param_diff = (prev_params[name] - param.data).pow(2)
            omega = omegas[name] / (param_diff + self.eps)
            ipt_groups[name] += omega

    def clear_omegas(self, reg_groups):
        omegas = reg_groups["omegas"]
        for name, omega in omegas.items():
            omegas[name] = torch.zeros_like(omega)


class RWalk:
    def reserve_pg(self, model, reg_groups):
        grad_pools = reg_groups["grad_pools"]
        param_pools = reg_groups["param_pools"]
        for name, param in model.named_parameters():
            if name not in grad_pools:
                continue
            grad_pools[name].append(deepcopy(param.grad))
            param_pools[name].append(param.clone().detach())

    def update_scores(self, reg_groups):
        scores = reg_groups["scores"]
        grad_pools = reg_groups["grad_pools"]
        param_pools = reg_groups["param_pools"]
        fim = reg_groups["fim"]
        eps = torch.finfo(torch.get_default_dtype()).eps
        for name in scores.keys():
            start_grad = grad_pools[name][0]
            param_diff = param_pools[name][-1] - param_pools[name][0]
            numerator = start_grad * param_diff
            denominator =  0.5 * fim[name] * (param_diff).pow(2) + eps
            scores[name] += (numerator / denominator).clamp(min=0)

    def update_fim(self, model, reg_groups):
        fim = reg_groups["fim"]
        alpha = 0.9
        for name, param in model.named_parameters():
            if name not in fim:
                continue
            fim[name] = alpha*param.grad.pow(2) + (1-alpha)*fim[name] 

    def calc_ipts(self, reg_groups):
        ipt_groups = reg_groups['ipt_groups']
        fim = reg_groups["fim"]
        scores = reg_groups["scores"]
        for name, ipt in ipt_groups.items():
            norm_fim = fim[name] / torch.norm(fim[name]).clamp(min=1e-12)
            norm_score = scores[name] / torch.norm(scores[name]).clamp(min=1e-12)
            ipt.copy_(norm_fim + 0.5*norm_score)


class NIS:
    def normalize_EWC(self, ipt_groups):
        for name, ipt in ipt_groups.items():
            out_channels = ipt.shape[0]
            flat_ipt = ipt.view(out_channels, -1)
            norm = torch.tile(flat_ipt.sum(dim=1).unsqueeze(1), (1, flat_ipt.shape[1]))
            if ipt.ndim > 1:
                in_channels = ipt.shape[1]
                norm /= in_channels
            flat_ipt.copy_(norm)


class reassignEWC:
    def adapt_ipts(self, model, reg_groups):
        ipt_groups = reg_groups['g']
        softmax = nn.Softmax(dim=1)
        for name, ipt in ipt_groups.items():
            curr_param = model.state_dict()[name]
            prev_param = reg_groups["init_params"][name]
            out_channels = curr_param.shape[0]
            param_diff = (curr_param - prev_param).pow(2).view(out_channels, -1)
            flat_ipt = ipt.view(out_channels, -1)
            norm = torch.tile(flat_ipt.sum(dim=1).unsqueeze(1), (1, flat_ipt.shape[1]))
            flat_ipt.copy_(norm * softmax(-1 * param_diff))


class NeuGrad:
    def calc_ipts(self, model, ipt_groups, dataloader, total_size):
        device = next(model.parameters()).device.type
        cur_size = len(dataloader)
        prev_size = total_size - cur_size

        # 正式开始采集
        rec_ngs = self.save_neuGrads(model, dataloader, ipt_groups)

        # 计算重要性
        for value in ipt_groups.values():
            value.mul_(prev_size)

        for name, ngs in rec_ngs.items():
            neuron_ipt = np.mean(np.concatenate(ngs), axis=0)
            in_features = ipt_groups[name + ".weight"].shape[1]
            if "fc" in name:
                w_ipt = torch.from_numpy(np.tile(neuron_ipt, (in_features,1)).T)
                b_ipt = torch.from_numpy(neuron_ipt)
            else:
                w, h = ipt_groups[name + ".weight"].shape[2:]
                channel_ipt = np.linalg.norm(neuron_ipt, axis=(1,2))
                w_ipt = np.expand_dims(channel_ipt, axis=(1,2,3))
                w_ipt = np.tile(w_ipt, (1,in_features,w,h))
                w_ipt = torch.from_numpy(w_ipt)
                b_ipt = torch.from_numpy(channel_ipt)
            ipt_groups[name + ".weight"] += w_ipt.to(device)
            ipt_groups[name + ".bias"] += b_ipt.to(device)

        for value in ipt_groups.values():
            value /= total_size
            value.abs_()

    def save_neuGrads(self, model, dataloader, ipt_groups):
        self.device = next(model.parameters()).device.type
        # 定义hook函数
        def bhook(module, grad_in, grad_out):
            nonlocal module_id, amount
            module_name = layers[module_id]
            rec_ngs[module_name].append(grad_out[0].clone().detach().cpu().numpy())
            module_id -= 1
            if module_id < 0:
                module_id = amount-1

        layers = self.register_layers(model, ipt_groups)
        rec_ngs = self.init_rec_ngs(layers)
        amount = len(layers)
        module_id = amount-1

        handles = self.add_handles(model, layers, bhook)

        print("Start capturing feature mapping.")
        print("------------------------------------------")
        self.pbar = tqdm(total=len(dataloader))
        self.dummy_loader = deepcopy(dataloader)
        self.core_algorithm(model)
        self.pbar.close()
        print("------------------------------------------")

        self.remove_handles(handles)
        return rec_ngs

    def core_algorithm(self, model):
        loss_fn = nn.CrossEntropyLoss()
        for batch, (X, y) in enumerate(self.dummy_loader):
            X, y = X.to(self.device), y.to(self.device)
            X.requires_grad = True
            # Compute prediction error
            pred = model(X)
            loss = loss_fn(pred, y)
            model.zero_grad()
            loss.backward()
            self.pbar.update(1)

    def register_layers(self, model, ipt_groups):
        layers = list()
        for name, param in model.named_parameters():
            if name not in ipt_groups:
                continue
            if "weight" not in name:
                continue
            layers.append(name.replace(".weight", ""))
        return layers

    def init_rec_ngs(self, layers):
        rec_ngs = dict()
        for layer in layers:
            rec_ngs[layer] = list()

        return rec_ngs

    def add_handles(self, model, layers, bhook):
        handles = []
        for name, module in model.named_modules():
            if name in layers:
                handles.append(module.register_full_backward_hook(bhook))
        return handles

    def remove_handles(self, handles):
        # 删除句柄
        for handle in handles:
            handle.remove()


class GradCAM(NeuGrad):
    def core_algorithm(self, model):
        for batch, (X, y) in enumerate(self.dummy_loader):
            X, y = X.to(self.device), y.to(self.device)
            X.requires_grad = True
            # Compute prediction error
            pred = model(X)
            mask = torch.zeros_like(pred).to(self.device)
            ones = torch.ones(X.shape[0], 1).to(self.device)
            mask.scatter_(1, y.unsqueeze(dim=1), ones)
            loss = (pred * mask).sum(dim=1).mean()
            model.zero_grad()
            loss.backward()
            self.pbar.update(1)


class NeuGradVar(NeuGrad):
    def calc_ipts(self, model, ipt_groups, dataloader, total_size):
        device = next(model.parameters()).device.type
        cur_size = len(dataloader)
        prev_size = total_size - cur_size

        # 正式开始采集
        rec_ngs = self.save_neuGrads(model, dataloader, ipt_groups)

        # 计算重要性
        for value in ipt_groups.values():
            value.mul_(prev_size)

        # 初始化重要性权重一阶矩和二阶矩
        moment1_ipts = dict()
        moment2_ipts = dict()
        for name, value in ipt_groups.items():
            moment1_ipts[name] = torch.zeros_like(value)
            moment2_ipts[name] = torch.zeros_like(value)

        for name, ngs in rec_ngs.items():
            neuron_ipt = np.mean(np.concatenate(ngs), axis=0)
            in_features = ipt_groups[name + ".weight"].shape[1]
            if "fc" in name:
                w_ipt = torch.from_numpy(np.tile(neuron_ipt, (in_features,1)).T)
                b_ipt = torch.from_numpy(neuron_ipt)
            else:
                w, h = ipt_groups[name + ".weight"].shape[2:]
                channel_ipt = np.linalg.norm(neuron_ipt, axis=(1,2))
                w_ipt = np.expand_dims(channel_ipt, axis=(1,2,3))
                w_ipt = np.tile(w_ipt, (1,in_features,w,h))
                w_ipt = torch.from_numpy(w_ipt)
                b_ipt = torch.from_numpy(channel_ipt)
            w_ipt = w_ipt.to(device)
            b_ipt = b_ipt.to(device)
            moment1_ipts[name + ".weight"] += w_ipt
            moment1_ipts[name + ".bias"] += b_ipt
            moment2_ipts[name + ".weight"] += w_ipt.pow(2)
            moment2_ipts[name + ".bias"] += b_ipt.pow(2)
            
            for name, value in ipt_groups.items():
                moment1_ipts[name].div_(cur_size)
                moment2_ipts[name].div_(cur_size)
                var = moment2_ipts[name]-moment1_ipts[name].pow(2)
                std = var.sqrt()
                value.add_(std.mul_(cur_size))

        for value in ipt_groups.values():
            value /= total_size
            value.abs_()


class interpEWC(Neuron):
    def calc_ipts(self, model, dataloader, loss_fn, ipt_groups):
        device = next(model.parameters()).device.type
        cur_size = len(dataloader)

        rec_fms, cur_fisher = self.save_neurons_and_calc_fisher(model, dataloader, loss_fn, ipt_groups)

        prev_fisher = {}
        for key, ipt in ipt_groups.items():
            prev_fisher[key] = ipt.clone().detach()
            ipt.zero_()

        for name, fms in rec_fms.items():
            neuron_ipt = np.mean(np.concatenate(fms), axis=0)
            in_features = ipt_groups[name + ".weight"].shape[1]
            if "fc" in name:
                w_ipt = torch.from_numpy(np.tile(neuron_ipt, (in_features,1)).T)
                b_ipt = torch.from_numpy(neuron_ipt)
            else:
                w, h = ipt_groups[name + ".weight"].shape[2:]
                channel_ipt = np.linalg.norm(neuron_ipt, axis=(1,2))
                w_ipt = np.expand_dims(channel_ipt, axis=(1,2,3))
                w_ipt = np.tile(w_ipt, (1,in_features,w,h))
                w_ipt = torch.from_numpy(w_ipt)
                b_ipt = torch.from_numpy(channel_ipt)
            w_ipt = torch.clamp(w_ipt.to(device), min=0)
            b_ipt = torch.clamp(b_ipt.to(device), min=0)
            # 归一化
            ##############################################################
            out_channels = w_ipt.shape[0]
            norm_w = torch.norm(w_ipt.view(out_channels, -1)[:, 0])+1e-12
            w_ipt /= norm_w

            norm_b = torch.norm(b_ipt)+1e-12
            b_ipt /= norm_b
            ##############################################################
            ipt_groups[name+".weight"] = cur_fisher[name+'.weight'] * w_ipt
            ipt_groups[name+".bias"] = cur_fisher[name+'.bias'] * b_ipt

        for name, ipt in ipt_groups.items():
            ipt.div_(cur_size)
            ipt.add_(prev_fisher[name])

    def save_neurons_and_calc_fisher(self, model, dataloader, loss_fn, ipt_groups):
        device = next(model.parameters()).device.type

        def fhook(module, f_in, f_out):
            nonlocal module_id, amount
            module_name = layers[module_id]
            rec_fms[module_name].append(f_out.clone().detach().cpu().numpy())
            module_id += 1
            if module_id >= amount:
                module_id = 0

        layers = self.register_layers(model, ipt_groups)
        rec_fms = self.init_rec_fms(layers)
        module_id = 0
        amount = len(layers)

        handles = self.add_handles(model, layers, fhook)
        cur_fisher = {}
        for key, ipt in ipt_groups.items():
            cur_fisher[key] = torch.zeros_like(ipt)

        print("Start capturing feature mapping.")
        print("------------------------------------------")
        pbar = tqdm(total=len(dataloader))
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            model.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if name not in ipt_groups:
                    continue
                cur_fisher[name] += param.grad.pow(2)
            pbar.update(1)
        pbar.close()
        print("------------------------------------------")

        self.remove_handles(handles)
        return rec_fms, cur_fisher


class pseudoNPC:
    def calc_ipts(self, model, ipt_groups, dataloader, total_size):
        device = next(model.parameters()).device.type
        cur_size = len(dataloader)
        prev_size = total_size - cur_size

        rec_fms, rec_ngs = self.save_neurons_and_neuGrads(model, dataloader, ipt_groups)

        for value in ipt_groups.values():
            value.mul_(prev_size)

        for name, fms in rec_fms.items():
            all_neuron = np.concatenate(fms)
            all_neuGrad = np.concatenate(rec_ngs[name])
            npc_ipt = np.mean(np.abs(all_neuron * all_neuGrad), axis=0)
            npc_ipt /= (np.sum(npc_ipt) / npc_ipt.size)
            in_features = ipt_groups[name + ".weight"].shape[1]
            if "fc" in name:
                w_ipt = torch.from_numpy(np.tile(npc_ipt, (in_features,1)).T)
                b_ipt = torch.from_numpy(npc_ipt)
            else:
                w, h = ipt_groups[name + ".weight"].shape[2:]
                channel_ipt = np.linalg.norm(npc_ipt, axis=(1,2))
                w_ipt = np.expand_dims(channel_ipt, axis=(1,2,3))
                w_ipt = np.tile(w_ipt, (1,in_features,w,h))
                w_ipt = torch.from_numpy(w_ipt)
                b_ipt = torch.from_numpy(channel_ipt)
            ipt_groups[name + ".weight"] += w_ipt.to(device)
            ipt_groups[name + ".bias"] += b_ipt.to(device)

        for value in ipt_groups.values():
            value /= total_size
            value.abs_()

    def save_neurons_and_neuGrads(self, model, dataloader, ipt_groups):
        device = next(model.parameters()).device.type

        # 定义fhook函数
        def fhook(module, f_in, f_out):
            nonlocal module_id, amount
            module_name = layers[module_id]
            rec_fms[module_name].append(f_out.clone().detach().cpu().numpy())
            module_id += 1
            if module_id >= amount:
                module_id = 0

        # 定义bhook函数
        def bhook(module, grad_in, grad_out):
            nonlocal rev_module_id, amount
            module_name = layers[rev_module_id]
            rec_ngs[module_name].append(grad_out[0].clone().detach().cpu().numpy())
            rev_module_id -= 1
            if rev_module_id < 0:
                rev_module_id = amount - 1

        layers = self.register_layers(model, ipt_groups)
        rec_fms = self.init_rec_fms(layers)
        rec_ngs = self.init_rec_ngs(layers)
        amount = len(layers)
        module_id = 0
        rev_module_id = amount - 1

        handles = self.add_handles(model, layers, fhook, bhook)

        loss_fn = nn.CrossEntropyLoss()
        print("Start capturing feature mapping.")
        print("------------------------------------------")
        pbar = tqdm(total=len(dataloader))
        for (X, y) in dataloader:
            X, y = X.to(device), y.to(device)
            X.requires_grad = True
            pred = model(X)
            loss = loss_fn(pred, y)
            model.zero_grad()
            loss.backward()
            pbar.update(1)
        pbar.close()
        print("------------------------------------------")

        self.remove_handles(handles)
        return rec_fms, rec_ngs

    def register_layers(self, model, ipt_groups):
        layers = list()
        for name, param in model.named_parameters():
            if name not in ipt_groups:
                continue
            if "weight" not in name:
                continue
            layers.append(name.replace(".weight", ""))
        return layers

    def init_rec_fms(self, layers):
        rec_fms = dict()
        for layer in layers:
            rec_fms[layer] = list()
        return rec_fms

    def init_rec_ngs(self, layers):
        rec_ngs = dict()
        for layer in layers:
            rec_ngs[layer] = list()
        return rec_ngs

    def add_handles(self, model, layers, fhook, bhook):
        handles = []
        for name, module in model.named_modules():
            if name in layers:
                handles.append(module.register_forward_hook(fhook))
                handles.append(module.register_full_backward_hook(bhook))
        return handles

    def remove_handles(self, handles):
        # 删除句柄
        for handle in handles:
            handle.remove()
