# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 15:28:45 2023

@author: admin
"""

import pickle
from copy import deepcopy
from collections import OrderedDict, defaultdict
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from agent import AgentController
from analyze import plot_multi_tasks_acc_2
from analyze import plot_parameter_hist


class CustomArgs:
    def __init__(self, experiment, config):
        for name, value in config[experiment].items():
            setattr(self, name, value)


def generate_agent(admin, appr):
    admin.register_approach(appr)
    agent = getattr(admin, appr)
    agent.reset()
    return agent

##############################################################################
class noiseMLP2(nn.Module):
    def __init__(self):
        super(noiseMLP2, self).__init__()
        self.flatten = nn.Flatten()
        self.backbone = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(28*28, 256)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(256, 256)),
            ("relu2", nn.ReLU()),
        ]))
        self.clf = nn.Linear(256, 2)
        self.fms = list()

    def forward(self, x):
        torch.manual_seed(self.noise)
        x = self.flatten(x)
        x = self.backbone(x)
        x += self.noise * torch.rand_like(x)
        self.fms.append(deepcopy(x))
        x = self.clf(x)
        return x

    def set_noise(self, noise):
        self.noise = noise


##############################################################################
class simpMLP2(nn.Module):
    def __init__(self, seed):
        super(simpMLP2, self).__init__()
        self.seed = seed
        self.flatten = nn.Flatten()
        self.bks = nn.ModuleList()
        self.dfs = nn.ModuleList()
        for task_id in range(5):
            backbone = nn.Sequential(OrderedDict([
                ("fc1", nn.Linear(28*28, 256)),
                ("relu1", nn.ReLU()),
                ("fc2", nn.Linear(256, 256)),
                ("relu2", nn.ReLU()),
            ]))
            self.bks.append(backbone)
            self.dfs.append(nn.Linear(256, 1))

    def forward(self, x, task_id):
        x = self.flatten(x)
        x = self.bks[task_id](x)
        x += self.noise
        x = self.dfs[task_id](x)
        return x

    def set_noise(self, noise):
        torch.manual_seed(noise+self.seed)
        self.noise = noise * torch.rand(1, 256)


##############################################################################
class PeelLayers:
    def __init__(self, experiment, repo_path):
        with open ("config.yaml", "r") as f:
            self.experiment = experiment
            self.config = yaml.load(f, Loader=yaml.FullLoader)
            self.set_admin()
            self.set_repo_path(repo_path)

    def set_admin(self):
        experiment = self.experiment
        args = CustomArgs(experiment, self.config)
        admin = AgentController(experiment, args)
        self.admin = admin
        self.train_seq = admin.data_seq["train"]
        self.test_seq = admin.data_seq["test"]
        self.reg_spec_seq = admin.data_seq["reg_spec"]

    def set_repo_path(self, repo_path):
        self.repo_path = repo_path

    def parameter_hist(self, task_id, layer, bins):
        def load_model(agent, task_id, repo_path):
            with open(repo_path, "rb") as f:
                model_repo = pickle.load(f)
            param_dict = model_repo[agent.appr][task_id].state_dict()
            with torch.no_grad():
                for name, param in agent.model.named_parameters():
                    dst_param = param_dict[name]
                    param.copy_(dst_param)

        appr = "baseline"
        agent = generate_agent(self.admin, appr)
        load_model(agent, task_id, self.repo_path)
        plot_parameter_hist(bins, agent.model, layer)

    def reorder_tasks(self, new_order):
        rt_instance = ReorderTasks(self.experiment, self.admin, self.train_seq,
                                   self.test_seq, self.reg_spec_seq)
        return rt_instance

    def assess_src_model_robustness(self):
        repo_path = self.repo_path
        admin = self.admin
        test_seq = self.test_seq
        asmr_instance = AssessSrcModelRobustness(repo_path, admin, test_seq)
        return asmr_instance

    def assess_simp_model_robustness(self, seed):
        repo_path = self.repo_path
        admin = self.admin
        test_seq = self.test_seq
        aspmr_instance = AssessSimpModelRobustness(repo_path, admin, test_seq, seed)
        return aspmr_instance

    def track_simp_model_trajectory(self):
        repo_path = self.repo_path
        admin = self.admin
        test_seq = self.test_seq
        tsmt_instance = TrackSimpModelTrajectory(repo_path, admin, test_seq)
        return tsmt_instance

    def analyze_error_signal_and_feature_mapping(self):
        repo_path = self.repo_path
        admin = self.admin
        train_seq = self.train_seq
        test_seq = self.test_seq
        reg_spec_seq = self.reg_spec_seq
        aesafm_instance = AnalyzeESandFM(repo_path, admin, train_seq, test_seq, reg_spec_seq)
        return aesafm_instance

    def analyze_image_importance(self):
        repo_path = self.repo_path
        admin = self.admin
        train_seq = self.train_seq
        test_seq = self.test_seq
        reg_spec_seq = self.reg_spec_seq
        aii_instance = AnalyzeImageImportance(repo_path, admin, train_seq, test_seq, reg_spec_seq)
        return aii_instance

    def analyze_gram_matrix(self):
        repo_path = self.repo_path
        admin = self.admin
        train_seq = self.train_seq
        test_seq = self.test_seq
        reg_spec_seq = self.reg_spec_seq
        agm_instance = AnalyzeGramMatrix(repo_path, admin, train_seq, test_seq, reg_spec_seq)
        return agm_instance


##############################################################################
class ReorderTasks:
    def __init__(self, experiment, admin, train_seq, test_seq, reg_spec_seq):
        self.experiment = experiment
        self.admin = admin
        self.train_seq = train_seq
        self.test_seq = test_seq
        self.reg_spec_seq = reg_spec_seq

    def train(self, new_order):
        appr = "baseline"
        new_agent = generate_agent(self.admin, appr)
        self.appr = appr
        self.new_agent = new_agent
        self.new_order = new_order
        train_seq = self.train_seq
        test_seq = self.test_seq
        reg_spec_seq = self.reg_spec_seq

        new_train_seq = list()
        new_test_seq = list()
        new_reg_spec_seq = list()
        for task_id in new_order:
            new_train_seq.append(train_seq[task_id])
            new_test_seq.append(test_seq[task_id])
            new_reg_spec_seq.append(reg_spec_seq[task_id])

        for task_id in range(new_agent.tasks):
            new_agent.run_one_task(new_train_seq, new_test_seq, new_reg_spec_seq, task_id)

    def visualize(self):
        experiment = self.experiment
        appr = self.appr
        new_agent = self.new_agent
        new_order = self.new_order
        all_test_acc = dict()
        all_test_acc[appr] = new_agent.test_acc
        plot_multi_tasks_acc_2(5, all_test_acc, experiment, new_order)


##############################################################################
class AssessSrcModelRobustness:
    def __init__(self, repo_path, admin, test_seq):
        self.noise_model = noiseMLP2()
        self.repo_path = repo_path
        self.admin = admin
        self.test_seq = test_seq

    def test(self, model, dataloader, loss_fn):
        device = next(model.parameters()).device.type
        size = len(dataloader.sampler)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        acc = correct / size
        print(f"Test Error: \n Accuracy: {(100*acc):>0.2f}%, Avg loss: {test_loss:>8f} \n")
        return int(correct), size

    def transfer_model(self, agent, task_id, repo_path, noise_model):
        with open(repo_path, "rb") as f:
            model_repo = pickle.load(f)
        param_dict = model_repo[agent.appr][task_id].state_dict()
        with torch.no_grad():
            for name, param in noise_model.named_parameters():
                dst_param = param_dict[name]
                param.copy_(dst_param)

    def run(self):
        appr = "baseline"
        admin = self.admin
        agent = generate_agent(admin, appr)
        repo_path = self.repo_path
        noise_model = self.noise_model
        test_seq = self.test_seq
        for task_id in range(5):
            self.transfer_model(agent, task_id, repo_path, noise_model)
            print(f"Task{task_id+1}")
            print("################################################")
            for noise_level in range(20):
                noise_model.set_noise(noise_level)
                self.test(noise_model, test_seq[task_id], agent.loss_fn)
            print("################################################")


##############################################################################
class AssessSimpModelRobustness:
    def __init__(self, repo_path, admin, test_seq, seed):
        self.seed = seed
        self.repo_path = repo_path
        self.admin = admin
        self.test_seq = test_seq
        self.appr = "baseline"
        with open(repo_path, "rb") as f:
            model_repo = pickle.load(f)
        self.model_repo = model_repo

    def make_equivalent_clf(self):
        appr = self.appr
        model_repo = self.model_repo
        dfs = {i:None for i in range(5)}
        for task_id, model in enumerate(model_repo[appr]):
            for name, param in model.named_parameters():
                param_dict = model.state_dict()
                dfs[task_id] = nn.Linear(256, 1)
                with torch.no_grad():
                    w1, w2 = param_dict["clf.weight"]
                    b1, b2 = param_dict["clf.bias"]
                    dfs[task_id].weight.copy_(w1-w2)
                    dfs[task_id].bias.copy_(b1-b2)
        self.dfs = dfs

    def simplify_model(self):
        dfs = self.dfs
        appr = self.appr
        model_repo = self.model_repo
        simp_model = simpMLP2(self.seed)
        with torch.no_grad():
            for task_id in range(5):
                param_dict = model_repo[appr][task_id].state_dict()
                for name, param in simp_model.bks[task_id].named_parameters():
                    dst_param = param_dict["backbone."+name]
                    param.copy_(dst_param)
            for task_id in range(5):
                simp_model.dfs[task_id].weight.copy_(dfs[task_id].weight)
                simp_model.dfs[task_id].bias.copy_(dfs[task_id].bias)
        self.simp_model = simp_model

    def test(self, model, dataloader, task_id):
        device = next(model.parameters()).device.type
        size = len(dataloader.sampler)
        model.eval()
        correct = 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X, task_id)
                quan_pred = torch.ones_like(pred, dtype=int)
                quan_pred[pred > 0] = 0
                quan_pred = quan_pred[:,0]
                correct += (quan_pred == y).type(torch.float).sum().item()
        acc = correct / size
        print(f"Test Error: \n  Accuracy: {(100*acc):>0.2f}%\n")
        return acc

    def run(self):
        self.make_equivalent_clf()
        self.simplify_model()

        simp_model = self.simp_model
        test_seq = self.test_seq
        for task_id in range(5):
            test_dataloader = test_seq[task_id]
            print(f"Task{task_id+1}")
            print("################################################")
            for noise_level in range(0, 11, 10):
                simp_model.set_noise(noise_level)
                self.test(simp_model, test_dataloader, task_id)
            print("################################################")


##############################################################################
class TrackSimpModelTrajectory(AssessSimpModelRobustness):
    def __init__(self, repo_path, admin, test_seq):
        with open(repo_path, "rb") as f:
            model_repo = pickle.load(f)
        self.model_repo = model_repo
        self.noise_model = noiseMLP2()
        self.repo_path = repo_path
        self.admin = admin
        self.test_seq = test_seq
        self.appr = "baseline"
        self.pred_recorder = defaultdict(lambda: defaultdict(list))
        self.acc_recorder = defaultdict(list)

    def test(self, model, dataloader, task_id):
        pred_recorder = self.pred_recorder
        device = next(model.parameters()).device.type
        size = len(dataloader.sampler)
        model.eval()
        correct = 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(dataloader):
                X, y = X.to(device), y.to(device)
                pred = model(X, task_id)
                pred_recorder[task_id][batch].append(pred)
                quan_pred = torch.ones_like(pred, dtype=int)
                quan_pred[pred > 0] = 0
                quan_pred = quan_pred[:,0]
                correct += (quan_pred == y).type(torch.float).sum().item()
        acc = correct / size
        print(f"Test Error: \n  Accuracy: {(100*acc):>0.2f}%\n")
        return int(correct), size

    def run(self):
        self.make_equivalent_clf()
        self.simplify_model()

        simp_model = self.simp_model
        test_seq = self.test_seq
        acc_recorder = self.acc_recorder

        for task_id in range(5):
            test_dataloader = test_seq[task_id]
            print(f"Task{task_id+1}")
            print("################################################")
            for noise_level in range(0, 11, 10):
                simp_model.set_noise(noise_level)
                result = self.test(simp_model, test_dataloader, task_id)
                acc_recorder[task_id].append(result[0]/result[1])
            print("################################################")

    def visualize(self):
        for task_id in range(5):
            y = self.acc_recorder[task_id]
            x = np.arange(len(y))
            plt.plot(x, y, label=task_id)
        plt.legend()
        plt.grid()
        plt.show()

    def print_characteristics(self):
        trend = {task_id: [] for task_id in range(5)}
        for task_id in range(5):
            for pair in self.pred_recorder[task_id].values():
                # pair的长度与噪声个数相等
                for a, b in zip(pair[0], pair[1]):
                    trend[task_id].append((a.item(), b.item()))

        # for pair in trend[1]:
        #     print(pair)

        pred_trajs = dict()
        for task_id in range(5):
            num = len(self.pred_recorder[task_id][0])
            temp = {j: [] for j in range(num)}
            for pair in self.pred_recorder[task_id].values():
                for j in range(num):
                    temp[j].append(pair[j].numpy())
            trash = list()
            for j in range(num):
                temp[j] = np.concatenate(temp[j], axis=0)
                trash.append(np.array(temp[j]))
            pred_trajs[task_id] = np.concatenate(trash, axis=1)

        stats = {i: [] for i in range(5)}
        for task_id in range(5):
            for j in range(2):
                temp = pred_trajs[task_id][:,j]
                stats[task_id].append((temp>=0).sum() / temp.shape[0])
        for task_id, stat in stats.items():
            print(f"{task_id} - prediction postive rate: {stat}")

        traits = {i: dict() for i in range(5)}
        for task_id in range(5):
            # 加噪前
            before = pred_trajs[task_id][:,0]
            pos_mask = before >= 0
            neg_mask = before < 0
            traits[task_id]['mean_pred'] = [(before*pos_mask).sum() / pos_mask.sum(),
                                            (before*neg_mask).sum() / neg_mask.sum()]
            after = pred_trajs[task_id][:,1]
            traits[task_id]['deviation'] = np.abs(after-before).mean()

        for task_id, trait in traits.items():
            print(f"{task_id}: {trait}")

        self.trend = trend
        self.stats = stats
        self.traits = traits
        self.pred_trajs = pred_trajs

    def mean_output_noise(self):
        simp_model = self.simp_model
        deviation = defaultdict(list)
        for j in range(16):
            torch.manual_seed(j)
            noise = 10 * torch.rand(1, 256)
            for task_id in range(5):
                pred = simp_model.dfs[task_id](noise)
                deviation[task_id].append(pred.item())
        mean_deviation = dict(deviation)
        for task_id in range(5):
            mean_deviation[task_id] =  sum(deviation[task_id]) / len(deviation[task_id])
        print(f"output noise: {mean_deviation}")

        self.mean_deviation = mean_deviation

    def clf_impact(self):
        simp_model = self.simp_model
        indep_dfs = list()
        for task_id in range(5):
            indep_dfs.append(deepcopy(simp_model.dfs[task_id]))

        impact = list()
        for interval in range(1, 256):
            cands = np.random.choice(256-interval, 256-interval, replace=False)
            temp = list()
            for start in cands:
                torch.manual_seed(10)
                noise = 10 * torch.rand(1, 256)
                # 嵌入零元素
                noise[0][start:start+interval] = 0
                pred = indep_dfs[1](noise)
                temp.append(pred.item())
            impact.append(sum(temp) / len(temp))
            print(f"interval: {interval}, impact: {sum(temp)/len(temp)}")

        y = impact
        x = np.arange(len(impact))
        plt.plot(x, y)
        plt.grid()
        plt.show()

        self.impact = impact

    def curr_output_noise(self):
        for task_id in range(5):
            pred = self.simp_model.dfs[task_id](self.simp_model.noise)
            print(pred)


##############################################################################
class AnalyzeESandFM:
    def __init__(self, repo_path, admin, train_seq, test_seq, reg_spec_seq):
        self.repo_path = repo_path
        self.admin = admin
        with open(repo_path, "rb") as f:
            model_repo = pickle.load(f)
        self.model_repo = model_repo
        self.train_seq = train_seq
        self.test_seq = test_seq
        self.reg_spec_seq = reg_spec_seq

    def reset_model(self, agent, task_id, model_repo, appr):
        torch.manual_seed(agent.args.classifier_seed+task_id-1)
        agent.model = agent._Model(agent.shared_weights).to(agent.device)
        pretrained_params = model_repo[appr][task_id-1].state_dict()
        with torch.no_grad():
            for name, param in agent.model.named_parameters():
                if "clf" in name:
                    break
                param.copy_(pretrained_params[name])

    def add_handles(self, agent):
        def bhook(module, grad_in, grad_out):
            self.rec_grad.append(grad_out[0].clone().detach().numpy())
        def fhook(module, f_in, f_out):
            self.rec_fm.append(f_in[0].clone().detach().numpy())

        handles = list()
        handles.append(agent.model.clf.register_full_backward_hook(bhook))
        handles.append(agent.model.clf.register_forward_hook(fhook))
        return handles

    def remove_handles(self, handles):
        for handle in handles:
            handle.remove()

    def customize_train(self, agent, train_seq, test_seq, reg_spec_seq, task_id):
        train_dataloader = train_seq[task_id]
        optimizer = agent._get_optim(agent.model.parameters())

        print("Start Task-{}:".format(task_id+1))
        for t in range(agent.args.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            agent._train(agent.model, train_dataloader, optimizer, agent.loss_fn, agent.reg_groups)
        print("Task-{} Done!".format(task_id+1))

    def run_all_tasks(self):
        appr = "baseline"
        admin = self.admin
        new_agent = generate_agent(admin, appr)
        self.new_agent = new_agent
        train_seq = self.train_seq
        test_seq = self.test_seq
        reg_spec_seq = self.reg_spec_seq

        self.fm_groups = list()
        self.grad_groups = list()
        for task_id in range(5):
            handles = self.add_handles(new_agent)
            self.rec_grad = list()
            self.rec_fm = list()
            self.customize_train(new_agent, train_seq, test_seq, reg_spec_seq, task_id)
            self.remove_handles(handles)
            new_agent._replace_clf(task_id)
            self.fm_groups.append(deepcopy(self.rec_fm))
            self.grad_groups.append(deepcopy(self.rec_grad))
        del self.rec_fm, self.rec_grad

    def run_one_task(self):
        # 单独训练某一特定任务
        appr = "baseline"
        admin = self.admin
        train_seq = self.train_seq
        test_seq = self.test_seq
        reg_spec_seq = self.reg_spec_seq
        new_agent = generate_agent(admin, appr)
        task_id = 1
        self.reset_model(new_agent, task_id)
        self.customize_train(new_agent, train_seq, test_seq, reg_spec_seq, task_id)
        new_agent._replace_clf(task_id)

    def draw_trajectory(self):
        traj = list()
        for task_id in range(5):
            rec_fm = self.fm_groups[task_id]
            rec_grad = self.grad_groups[task_id]
            one_epoch = len(rec_grad) // self.new_agent.args.epochs
            fms = []
            grads = []
            temp = {'fm': [], 'grad': []}
            for idx, (fm, grad) in enumerate(zip(rec_fm, rec_grad)):
                fms.append(fm)
                grads.append(grad)
                if idx % one_epoch == one_epoch-1:
                    temp['fm'].append(np.concatenate(fms, axis=0))
                    temp['grad'].append(np.concatenate(grads, axis=0))
                    fms.clear()
                    grads.clear()
            traj.append(temp)
        return traj

    def calc_difference(self):
        fm_mag = {i: [] for i in range(5)}
        grad_mag = {i: [] for i in range(5)}
        traj = self.draw_trajectory()
        self.traj = traj
        for task_id in range(5):
            for j in range(self.new_agent.args.epochs):
                mean = traj[task_id]['fm'][j].mean()
                fm_mag[task_id].append(mean)

                # 这种做法只是单纯分类器1和分类器2相加
                # mean = traj[task_id]['grad'][j].mean()
                # 实验发现二分类任务分类器梯度之和为0，所以只取一个分类器的梯度即可
                observe_grad = traj[task_id]['grad'][j][:,0]
                mean = np.abs(observe_grad).mean()
                # 重点应该看的是分类器之间的差异度
                # clf1_grad = traj[task_id]['grad'][j][:,0]
                # clf2_grad = traj[task_id]['grad'][j][:,1]
                # mean = (clf1_grad-clf2_grad).mean()
                grad_mag[task_id].append(mean)
        return fm_mag, grad_mag

    def visualize(self, save=False):
        fm_mag, grad_mag = self.calc_difference()
        self.fm_mag = fm_mag
        self.grad_mag = grad_mag
        for task_id, mag in fm_mag.items():
            y = mag
            x = np.arange(len(y))
            plt.plot(x, y, label=f"Task{task_id+1}")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.axis([-1, 41, 0.05, 0.85])
        plt.legend(ncol=2, fontsize=16)
        plt.grid()
        if save:
            plt.savefig('fm_traj.png', dpi=720)
        plt.show()

        for task_id, mag in grad_mag.items():
            y = mag
            x = np.arange(len(y))
            plt.plot(x, y, label=f"Task{task_id+1}")
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.yscale('log')
        plt.legend(fontsize=16)
        plt.grid()
        if save:
            plt.savefig('grad_traj.png', dpi=720)
        plt.show()

    def custom_config(self, backbone_seed, clf_seed):
        self.admin.args.shared_weights_seed = backbone_seed
        self.admin.args.classifier_seed = clf_seed
        src_path = self.admin.args.model_path.replace(".pth", '')
        self.admin.args.model_path = f"{src_path}_bs{backbone_seed}cs{clf_seed}.pth"


##############################################################################
class AnalyzeImageImportance:
    def __init__(self, repo_path, admin, train_seq, test_seq, reg_spec_seq):
        self.repo_path = repo_path
        self.admin = admin
        with open(repo_path, "rb") as f:
            model_repo = pickle.load(f)
        self.model_repo = model_repo
        self.train_seq = train_seq
        self.test_seq = test_seq
        self.reg_spec_seq = reg_spec_seq

    def add_handles(self, agent, layer, add_fhook, add_bhook):
        def bhook(module, grad_in, grad_out):
            self.rec_grad.append(grad_in[0].clone().detach().numpy())
        def fhook(module, f_in, f_out):
            self.rec_fm.append(f_in[0].clone().detach().numpy())

        handles = list()
        if add_fhook:
            for name, module in agent.model.named_modules():
                if name == layer:
                    handles.append(module.register_forward_hook(fhook))
        if add_bhook:
            for name, module in agent.model.named_modules():
                if name == layer:
                    handles.append(module.register_full_backward_hook(bhook))
        return handles

    def remove_handles(self, handles):
        for handle in handles:
            handle.remove()

    def customize_train(self, agent, train_seq, test_seq, reg_spec_seq, task_id):
        train_dataloader = train_seq[task_id]
        optimizer = agent._get_optim(agent.model.parameters())

        print("Start Task-{}:".format(task_id+1))
        for t in range(agent.args.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            agent._train(agent.model, train_dataloader, optimizer, agent.loss_fn, agent.reg_groups)
        print("Task-{} Done!".format(task_id+1))

    def collect_grads(self, layer, add_fhook, add_bhook):
        appr = "baseline"
        admin = self.admin
        new_agent = generate_agent(admin, appr)
        self.new_agent = new_agent
        train_seq = self.train_seq
        test_seq = self.test_seq
        reg_spec_seq = self.reg_spec_seq

        self.fm_groups = list()
        self.grad_groups = list()
        for task_id in range(5):
            handles = self.add_handles(new_agent, layer, add_fhook, add_bhook)
            self.rec_grad = list()
            self.rec_fm = list()
            self.customize_train(new_agent, train_seq, test_seq, reg_spec_seq, task_id)
            self.remove_handles(handles)
            new_agent._replace_clf(task_id)
            if add_fhook:
                self.fm_groups.append(deepcopy(self.rec_fm))
            if add_bhook:
                self.grad_groups.append(deepcopy(self.rec_grad))
        del self.rec_fm, self.rec_grad

    def calc_image_importance(self, layer, add_fhook, add_bhook):
        self.collect_grads(layer, add_fhook, add_bhook)
        trajs = dict()
        for task_id in range(5):
            rec_grad = self.grad_groups[task_id]
            one_epoch_num = len(rec_grad) // self.new_agent.args.epochs
            grads = []
            temp = []
            for idx, grad in enumerate(rec_grad):
                grads.append(grad)
                if idx % one_epoch_num == one_epoch_num-1:
                    temp.append(np.concatenate(grads, axis=0))
                    grads.clear()
            trajs[task_id] = temp
        image_importance = dict()
        for task_id, traj in trajs.items():
            s = 0
            for grad in traj:
                s += grad
            image_importance[task_id] = s
        self.trajs = trajs
        self.image_importance = image_importance


class AnalyzeGramMatrix():
    def __init__(self, repo_path, admin, train_seq, test_seq, reg_spec_seq):
        self.repo_path = repo_path
        self.admin = admin
        if repo_path:
            with open(repo_path, "rb") as f:
                model_repo = pickle.load(f)
            self.model_repo = model_repo
        self.train_seq = train_seq
        self.test_seq = test_seq
        self.reg_spec_seq = reg_spec_seq

    def add_handles(self, model, layers, add_fhook, add_bhook):
        def bhook(module, grad_in, grad_out):
            self.rec_grad.append(grad_in[0].clone().detach().numpy())
        def fhook(module, f_in, f_out):
            self.rec_fm.append(f_in[0].clone().detach().numpy())

        handles = list()
        if add_fhook:
            for name, module in model.named_modules():
                if name in layers:
                    handles.append(module.register_forward_hook(fhook))
        if add_bhook:
            for name, module in model.named_modules():
                if name in layers:
                    handles.append(module.register_full_backward_hook(bhook))
        return handles

    def remove_handles(self, handles):
        for handle in handles:
            handle.remove()

    def customize_train(self, agent, train_seq, test_seq, reg_spec_seq, task_id):
        train_dataloader = train_seq[task_id]
        optimizer = agent._get_optim(agent.model.parameters())

        print("Start Task-{}:".format(task_id+1))
        for t in range(agent.args.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            agent._train(agent.model, train_dataloader, optimizer, agent.loss_fn, agent.reg_groups)
        print("Task-{} Done!".format(task_id+1))

    def reset(self, backbone_seed, clf_seed, appr):
        admin = deepcopy(self.admin)
        admin.args.shared_weights_seed = backbone_seed
        admin.args.classifier_seed = clf_seed
        src_path = admin.args.model_path.replace(".pth", '')
        admin.args.model_path = f"{src_path}_bs{backbone_seed}cs{clf_seed}.pth"
        new_agent = generate_agent(admin, appr)
        self.new_agent = new_agent

    def collect_statics(self, layers, add_fhook, add_bhook):
        new_agent = self.new_agent
        train_seq = self.train_seq
        test_seq = self.test_seq
        reg_spec_seq = self.reg_spec_seq

        self.fm_groups = list()
        self.grad_groups = list()
        for task_id in range(5):
            handles = self.add_handles(new_agent.model, layers, add_fhook, add_bhook)
            self.rec_grad = list()
            self.rec_fm = list()
            self.customize_train(new_agent, train_seq, test_seq, reg_spec_seq, task_id)
            self.remove_handles(handles)
            new_agent._calc_ipts(task_id, reg_spec_seq)
            new_agent._replace_clf(task_id)
            if add_fhook:
                self.fm_groups.append(deepcopy(self.rec_fm))
            if add_bhook:
                self.grad_groups.append(deepcopy(self.rec_grad))
        del self.rec_fm, self.rec_grad

    def calc_gram_matrix(self, layers, is_grad=False):
        trajs = dict()
        for task_id in range(5):
            if is_grad:
                recorder = self.grad_groups[task_id]
            else:
                recorder = self.fm_groups[task_id]
            one_epoch_num = len(recorder) // self.new_agent.args.epochs
            stats = []
            temp = []
            for idx, data in enumerate(recorder):
                stats.append(data)
                if idx % one_epoch_num == one_epoch_num-1:
                    temp.append(deepcopy(stats))
                    stats.clear()
            trajs[task_id] = temp
        gram_matrix_trajs = dict()
        for task_id, traj in trajs.items():
            gram_matrix_trajs[task_id] = list()
            for one_epoch_stats in traj:
                temp = []
                for one_batch_stats in one_epoch_stats:
                    temp.append(one_batch_stats @ one_batch_stats.T)
                gram_matrix_trajs[task_id].append(temp)
        self.trajs = trajs
        self.gram_matrix_trajs = gram_matrix_trajs

    def calc_fms_similarity(self, task_id, src_time_id, dst_time_id):
        multi_fm_groups = self.multi_fm_groups
        a = multi_fm_groups[task_id][src_time_id]
        b = multi_fm_groups[task_id][dst_time_id]
        norm_a = np.linalg.norm(a, axis=1)
        norm_b= np.linalg.norm(b, axis=1)
        norm_a = np.reshape(norm_a, (-1, 1))
        norm_b = np.reshape(norm_b, (-1, 1))
        norm_ab = norm_a @ norm_b.T
        correl = a @ b.T
        correl /= norm_ab
        return correl

    def preprocess(self):
        multi_fm_groups = {i: [] for i in range(5)}
        for task_id in range(5):
            for fm_groups in self.multi_fm_groups[task_id]:
                all_fms = np.concatenate(fm_groups, axis=0)
                multi_fm_groups[task_id].append(all_fms)
        self.multi_fm_groups = multi_fm_groups

    def collect_multi_fms(self, appr, layers, add_fhook=True, add_bhook=False):
        test_seq = self.test_seq
        new_agent = self.new_agent
        models = self.model_repo[appr]
        self.multi_fm_groups = {i: [] for i in range(5)}
        for task_id in range(5):
            test_dataloader = test_seq[task_id]
            for j in range(5):
                model = models[j]
                self.rec_fm = list()
                handles = self.add_handles(model, layers, add_fhook, add_bhook)
                new_agent._test(model, test_dataloader, new_agent.loss_fn)
                self.remove_handles(handles)
                self.multi_fm_groups[task_id].append(deepcopy(self.rec_fm))
        del self.rec_fm

    def collect_labels(self):
        test_seq = self.test_seq
        label_inds = {i: {0: [], 1: []} for i in range(5)}
        for task_id in range(5):
            test_dataloader = test_seq[task_id]
            batch_size = test_dataloader.batch_size
            temp = {0: [], 1: []}
            for batch, (X, y) in enumerate(test_dataloader):
                label = y.cpu().numpy()
                for i in range(2):
                    relative_inds = np.where(label==i)[0]
                    absolute_inds = batch*batch_size + relative_inds
                    temp[i].append(absolute_inds)
            for i in range(2):
                label_inds[task_id][i] = np.concatenate(temp[i])
        return label_inds