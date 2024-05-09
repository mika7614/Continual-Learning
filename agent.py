import os
from copy import deepcopy
from collections import deque
import torch
from torch import nn
import utils
import approaches
import custom_models
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter('runs/Split5_MNIST')


class RawProcess:
    def __init__(self, args, Model):
        self.args = args
        self._Model = Model
        self.tasks = self.args.tasks
        self.loss_fn = nn.CrossEntropyLoss()
        self._train = utils.RawTrain().train
        self._test = utils.test
        self.reg_groups = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.appr = "baseline"

    def run(self, data_seq):
        self.reset()
        train_seq = data_seq["train"]
        test_seq = data_seq["test"]
        reg_spec_seq = data_seq["reg_spec"]
        # reg_spec_seq = data_seq["train"]

        for task_id in range(self.tasks):
            self.run_one_task(train_seq, test_seq, reg_spec_seq, task_id)

        print("All Tasks Completed.")

    def run_one_task(self, train_seq, test_seq, reg_spec_seq, task_id):
        train_dataloader = train_seq[task_id]
        test_dataloader = test_seq[task_id]
        optimizer = self._get_optim(self.model.parameters())

        print("Start Task-{}:".format(task_id+1))
        for t in range(self.args.epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loss = self._train(self.model, train_dataloader, optimizer, self.loss_fn, self.reg_groups)
            result = self._test(self.model, test_dataloader, self.loss_fn)
            # writer.add_scalar('Loss/train', train_loss, self.args.epochs * task_id + t)
            # writer.add_scalar('Loss/test', result[0], self.args.epochs * task_id + t)
            # writer.add_scalar('Accuracy/test', result[1], self.args.epochs * task_id + t)
            # writer.add_histogram('backbone.fc1.weight.grad', self.model.backbone.fc1.weight.grad, self.args.epochs * task_id + t)
            # for i in range(784):
            #     writer.add_histogram(f'fc1.weight.grad{[i]}', self.model.backbone.fc1.weight.grad[:,i], self.args.epochs * task_id + t)
        print("Task-{} Done!".format(task_id+1))

        self._record_acc(task_id, result, test_seq)
        self._save_model()
        self._calc_ipts(task_id, reg_spec_seq)
        self._save_ipts()
        self._save_prev_params()
        self._replace_clf(task_id)

    def reset(self):
        self._init_acc_tracker()
        self._init_model_tracker()
        self._init_ipt_tracker()
        self._generate_shared_weights()
        self._init_model()
        self._load_model()
        self._init_regs()
        # Online algorithms only
        self._init_extra_regs()

        # Ensure the repoductivity
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def _get_optim(self, params):
        lr = 1e-3
        self.lr = lr
        if "CIFAR100" in self.args.dataset:
            optimizer = torch.optim.Adam(params, lr, betas=(0.9, 0.999))
        else:
            optimizer = torch.optim.SGD(params, lr, momentum=0.9, nesterov=True)
        return optimizer

    def _init_acc_tracker(self):
        print("Initialize the accuracy tracker:", end=' ')
        self.test_acc = {i:[] for i in range(self.tasks)}
        print("Complete.")

    def _init_model_tracker(self):
        print("Initialize the model tracker:", end=' ')
        self.model_repo = list()
        self.backbone_repo = list()
        print("Complete.")

    def _generate_shared_weights(self):
        print("Generate the shared weights:", end=' ')
        self.shared_weights = dict()
        for name in self.args.shared_weights:
            torch.manual_seed(self.args.shared_weights_seed)
            self.shared_weights[name] = getattr(self._Model(), name)
        print("Complete.")

    def _init_model(self):
        print("Initialize the {} model:".format(self._Model.__name__), end=' ')
        torch.manual_seed(self.args.shared_weights_seed)
        self.model = self._Model(self.shared_weights).to(self.device)
        if self.tasks == 4:
            torch.manual_seed(self.args.shared_weights_seed)
            clf = nn.Linear(self.model.clf.in_features, 3)
            self.model.clf = clf.to(self.device)
        print("Complete.")

    def _load_model(self):
        model_path = self.args.model_path
        print("Load model from {}:".format(model_path), end=' ')
        if not os.path.exists(model_path):
            torch.save(self.model.state_dict(), model_path)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path))
            except:
                self.model.load_state_dict(torch.load(model_path,
                                                      map_location='cpu'))
        print("Complete.")

    def _replace_clf(self, task_id):
        print("Replace the classifier:", end=' ')
        torch.manual_seed(self.args.classifier_seed + task_id)
        self.model = self._Model(self.shared_weights).to(self.device)
        if self.tasks == 4:
            if task_id < 2:
                torch.manual_seed(self.args.classifier_seed + task_id)
                clf = nn.Linear(self.model.clf.in_features, 3)
                self.model.clf = clf.to(self.device)
        print("Complete.")

    def _record_acc(self, task_id, result, test_seq):
        print("Record the test accuracy:")
        self.test_acc[task_id].append(result)
        for j in range(task_id):
            temp_loader = test_seq[j]
            result = self._test(self.model_repo[j], temp_loader, self.loss_fn)
            self.test_acc[j].append(result)
        print("Complete.")

    def _save_model(self):
        print("Save the current model in the model repository:", end=' ')
        self.model_repo.append(self.model)
        self.backbone_repo.append(deepcopy(self.model))
        print("Complete.")

    def _init_ipt_tracker(self):
        pass

    def _save_ipts(self):
        pass

    def _save_prev_params(self):
        pass

    def _calc_ipts(self, task_id, reg_spec_seq):
        pass

    def _init_regs(self):
        pass

    def _init_extra_regs(self):
        pass

class RegProcess(RawProcess):
    def __init__(self, args, Model, appr):
        self.args = args
        self._Model = Model
        self.tasks = self.args.tasks
        self.loss_fn = nn.CrossEntropyLoss()
        self._train = utils.RegularTrain().train
        self._test = utils.test
        self.reg_groups = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.appr = appr

    def _save_prev_params(self):
        model = self.model
        reg_groups = self.reg_groups
        for name, param in model.named_parameters():
            reg_groups['prev_params'][name] = param.clone().detach()

    def _init_regs(self):
        def init_ipt_groups():
            ipt_groups = dict()
            for name, param in self.model.named_parameters():
                block_name = name.split('.')[0]
                if block_name not in self.args.shared_weights:
                    continue
                ipt_groups[name] = torch.zeros_like(param)
            return ipt_groups

        appr = self.appr
        self.reg_groups['ipt_groups'] = init_ipt_groups()
        self.reg_groups["prev_params"] = {}
        self._save_prev_params()
        self.reg_groups["coef"] = self.args.appr_hyp[appr]

    def _init_ipt_tracker(self):
        print("Initialize the importance weights tracker:", end=' ')
        self.ipts_repo = list()
        print("Complete.")

    def _save_ipts(self):
        print("Save the current importance weights:", end=' ')
        self.ipts_repo.append(deepcopy(self.reg_groups["ipt_groups"]))
        print("Complete.")


class EWC_Process(RegProcess):
    def __init__(self, args, Model):
        super(EWC_Process, self).__init__(args, Model, "EWC")
        self._Appr = approaches.EWC()

    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model
        dataloader = reg_spec_seq[task_id]
        loss_fn = self.loss_fn
        ipt_groups = self.reg_groups["ipt_groups"]

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, dataloader, loss_fn, ipt_groups)
        print("Complete.")


class MAS_Process(RegProcess):
    def __init__(self, args, Model):
        super(MAS_Process, self).__init__(args, Model, "MAS")
        self._Appr = approaches.MAS()
        self.total_size = 0

    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model
        dataloader = reg_spec_seq[task_id]
        self.total_size += len(dataloader)
        ipt_groups = self.reg_groups["ipt_groups"]

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, dataloader, ipt_groups, self.total_size)
        print("Complete.")


class SCP_Process(RegProcess):
    def __init__(self, args, Model):
        super(SCP_Process, self).__init__(args, Model, "SCP")
        self._Appr = approaches.SCP()

    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model
        dataloader = reg_spec_seq[task_id]
        ipt_groups = self.reg_groups["ipt_groups"]

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, dataloader, ipt_groups, task_id)
        print("Complete.")


class SI_Process(RegProcess):
    def __init__(self, args, Model):
        super(SI_Process, self).__init__(args, Model, "SI")
        self._Appr = approaches.SI()
        self._init_online_train()

    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, self.reg_groups)
        self._Appr.clear_omegas(self.reg_groups)
        print("Complete.")

    def _init_extra_regs(self):
        ipt_groups = self.reg_groups["ipt_groups"]
        self.reg_groups["omegas"] = deepcopy(ipt_groups)
        self.reg_groups["last_params"] = dict()
        self._Appr.update_last_params(self.model, self.reg_groups)

    def _init_online_train(self):
        class siTrain(utils.RegularTrain):
            def __init__(self, approach):
                self.approach = approach

            def update_reg_groups(self, model, reg_groups, batch=None):
                self.approach.update_omegas(model, reg_groups)
                self.approach.update_last_params(model, reg_groups)

        self._train = siTrain(self._Appr).train


class Rwalk_Process(RegProcess):
    def __init__(self, args, Model):
        super(Rwalk_Process, self).__init__(args, Model, "RWalk")
        self._Appr = approaches.RWalk()
        self._interval = 50
        self._init_online_train()

    def _calc_ipts(self, task_id, reg_spec_seq):
        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(self.reg_groups)
        print("Complete.")

    def _init_extra_regs(self):
        ipt_groups = self.reg_groups["ipt_groups"]
        param_pools = dict()
        grad_pools = dict()
        for name, param in ipt_groups.items():
            param_pools[name] = deque(maxlen=self._interval+1)
            grad_pools[name] = deque(maxlen=self._interval+1)
        self.reg_groups["param_pools"] = param_pools
        self.reg_groups["grad_pools"] = grad_pools
        self.reg_groups["scores"] = deepcopy(ipt_groups)
        self.reg_groups["fim"] = deepcopy(ipt_groups)

    def _init_online_train(self):
        class rwalkTrain(utils.RegularTrain):
            def __init__(self, approach, interval):
                self.approach = approach
                self.interval = interval

            def update_reg_groups(self, model, reg_groups, batch):
                self.approach.reserve_pg(model, reg_groups)
                self.approach.update_fim(model, reg_groups)
                if batch >= self.interval:
                    self.approach.update_scores(reg_groups)

        self._train = rwalkTrain(self._Appr, self._interval).train


class Neuron_Process(RegProcess):
    def __init__(self, args, Model):
        super(Neuron_Process, self).__init__(args, Model, "Neuron")
        self._Appr = approaches.Neuron()
        self.total_size = 0

    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model
        dataloader = reg_spec_seq[task_id]
        self.total_size += len(dataloader)
        ipt_groups = self.reg_groups["ipt_groups"]

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, ipt_groups, dataloader, self.total_size)
        print("Complete.")


class NeuGradProcess(RegProcess):
    def __init__(self, args, Model):
        super(NeuGradProcess, self).__init__(args, Model, "NeuGrad")
        self._Appr = approaches.NeuGrad()
        self.total_size = 0

    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model
        dataloader = reg_spec_seq[task_id]
        self.total_size += len(dataloader)
        ipt_groups = self.reg_groups["ipt_groups"]

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, ipt_groups, dataloader, self.total_size)
        print("Complete.")


class GradCAMProcess(RegProcess):
    def __init__(self, args, Model):
        super(GradCAMProcess, self).__init__(args, Model, "GradCAM")
        self._Appr = approaches.GradCAM()
        self.total_size = 0

    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model
        dataloader = reg_spec_seq[task_id]
        self.total_size += len(dataloader)
        ipt_groups = self.reg_groups["ipt_groups"]

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, ipt_groups, dataloader, self.total_size)
        print("Complete.")


class NeuGradVarProcess(RegProcess):
    def __init__(self, args, Model):
        super(NeuGradVarProcess, self).__init__(args, Model, "NeuGradVar")
        self._Appr = approaches.NeuGradVar()
        self.total_size = 0

    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model
        dataloader = reg_spec_seq[task_id]
        self.total_size += len(dataloader)
        ipt_groups = self.reg_groups["ipt_groups"]

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, ipt_groups, dataloader, self.total_size)
        print("Complete.")


class itpEWC_Process(RegProcess):
    def __init__(self, args, Model):
        super(itpEWC_Process, self).__init__(args, Model, "itpEWC")
        self._Appr = approaches.interpEWC()

    def _init_extra_regs(self):
        ipt_groups = self.reg_groups["ipt_groups"]
        self.reg_groups["src_ipts"] = deepcopy(ipt_groups)
        
    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model
        dataloader = reg_spec_seq[task_id]
        loss_fn = self.loss_fn
        src_ipts = self.reg_groups["src_ipts"]

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, dataloader, loss_fn, src_ipts)
        print("Complete.")
        self._clamp_ipts(task_id, reg_spec_seq)

    def _clamp_ipts(self, task_id, reg_spec_seq):
        src_ipts = self.reg_groups["src_ipts"]
        self.reg_groups["ipt_groups"] = deepcopy(src_ipts)
        ipt_groups = self.reg_groups["ipt_groups"]
        coef = self.reg_groups["coef"]
        lr = self.lr

        print("Start clamping important weights:", end=' ')
        for name, ipt in ipt_groups.items():
            clamp_ipt = torch.clamp(ipt, min=0, max=0.5/(coef*lr))
            # 重要性权重取值范围压缩
            ###################################
            # min_ipt = torch.min(clamp_ipt)
            # max_ipt = torch.max(clamp_ipt)
            # interval = max_ipt-min_ipt
            # if interval < 1e-12:
            #     continue
            if "conv" in name:
                # # 以卷积层为单位
                # m = 0.2
                # n = 0.2
                # min_target = min_ipt + m*interval
                # max_target = max_ipt - n*interval
                # k = (max_target - min_target) / interval
                # b = min_target - k*min_ipt
                # ipt_groups[name] = k*clamp_ipt + b

                # # 以输出神经元为单位
                m = 0.2
                n = 0.2
                out_channels = clamp_ipt.shape[0]
                reshape_ipt = clamp_ipt.reshape(out_channels, -1)
                min_ipt = torch.min(reshape_ipt, dim=1).values
                max_ipt = torch.max(reshape_ipt, dim=1).values
                interval = max_ipt-min_ipt
                min_target = min_ipt + m*interval
                max_target = max_ipt - n*interval
                k = (max_target - min_target) / (interval+1e-12)
                b = min_target - k*min_ipt

                if clamp_ipt.ndim > 1:
                    exp_dims = [1 for i in range(1, clamp_ipt.ndim)]
                    exp_dims.insert(0, -1)
                    k = k.reshape(exp_dims)
                    b = b.reshape(exp_dims)
                ipt_groups[name] = k*clamp_ipt + b

                # pass
            else:
                # # 两侧堆积
                # mid_ipt = (min_ipt+max_ipt) / 2
                # mag = 8 / interval
                # dst_ipt = interval * nn.Sigmoid()(mag*(clamp_ipt-mid_ipt)) + min_ipt
                # ipt_groups[name] = dst_ipt
                # # 归一化
                # srcVol = clamp_ipt.sum()
                # dstVol = dst_ipt.sum()
                # ipt_groups[name] *= (srcVol/dstVol)

                # # 高幅截断
                # thresh = torch.quantile(clamp_ipt, q=0.999)
                # clamp_ipt[clamp_ipt > thresh] = 0
                # ipt_groups[name] = clamp_ipt

                # # 平滑
                # m = 0.2
                # n = 0.2
                # min_target = min_ipt + m*interval
                # max_target = max_ipt - n*interval
                # k = (max_target - min_target) / interval
                # b = min_target - k*min_ipt
                # ipt_groups[name] = k*clamp_ipt + b

                pass
            ###################################
        print("Complete.")


class pseudoNPC_Process(RegProcess):
    def __init__(self, args, Model):
        super(pseudoNPC_Process, self).__init__(args, Model, "pseudoNPC")
        self._Appr = approaches.pseudoNPC()
        self.total_size = 0

    def _calc_ipts(self, task_id, reg_spec_seq):
        model = self.model
        dataloader = reg_spec_seq[task_id]
        self.total_size += len(dataloader)
        ipt_groups = self.reg_groups["ipt_groups"]

        print("Calculate the {} importance:".format(self.appr), end=' ')
        self._Appr.calc_ipts(model, ipt_groups, dataloader, self.total_size)
        print("Complete.")


class FreezeClfProcess(RawProcess):
    def __init__(self, args, Model):
        super(FreezeClfProcess, self).__init__(args, Model)
        self._train = self.freeze_clf_train()

    def freeze_clf_train(self):
        class NewRawTrain(utils.RawTrain):
            def base_train(self, model, X, y, optimizer,
                       loss_fn, reg_groups, batch):
                device = next(model.parameters()).device.type
                lr = optimizer.state_dict()["param_groups"][0]['lr']

                X, y = X.to(device), y.to(device)
                # Compute prediction error
                pred = model(X)
                # Calculate the loss
                regular = self.calc_reg(model, reg_groups, lr)
                loss = loss_fn(pred, y) + regular
                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                # freeze the classifier
                self.freeze_layer(model)
                optimizer.step()
                # Online algorithms only
                self.update_reg_groups(model, reg_groups, batch)
                return loss
        return NewRawTrain().train


class ReorderFreezeClfProcess(FreezeClfProcess):
    def __init__(self, args, Model):
        super(ReorderFreezeClfProcess, self).__init__(args, Model)
        self.clf_sample = None

    def _calc_ipts(self, task_id, reg_spec_seq):
        self.src_vec = self.model.clf.weight[0] - self.model.clf.weight[1]

    def aux_alg_inds(self, src_vec, clf_diff):
        a = clf_diff.clone()
        b = src_vec.clone()
        n = len(b)
        groups = {(i,j): [] for i in range(-1, 2, 2) for j in range(-1, 2, 2)}
        for i in range(n):
            sign_a = np.sign(a[i].item())
            sign_b = np.sign(b[i].item())
            groups[(sign_b, sign_a)].append(i)
        dst_inds = np.arange(n, dtype=int)
        count = 0
        same_ind = 0
        diff_ind = 0
        same_limit = min(len(groups[1,1]), len(groups[-1,-1]))
        diff_limit = min(len(groups[1,-1]), len(groups[-1,1]))
        sample_a = a.clone()
        solutions = list()
        while count < n:
            if same_ind >= same_limit or diff_ind >= diff_limit:
                break
            if a @ b < 0:
                i = groups[(1,-1)][diff_ind]
                j = groups[(-1,1)][diff_ind]
                diff_ind += 1
            else:
                i = groups[(1,1)][same_ind]
                j = groups[(-1,-1)][same_ind]
                same_ind += 1
            dst_inds[i] = j
            dst_inds[j] = i
            a = sample_a[dst_inds]
            c = (a @ b).item()
            print(c)
            solutions.append((a.clone(), c, deepcopy(dst_inds)))
            count += 1
        if solutions:
            a, c, dst_inds = min(solutions, key=lambda x: abs(x[1]))
        print(f'result = {c}')
        return dst_inds

    def permute_clf(self, src_vec, model):
        src_vec = torch.squeeze(src_vec)
        clf_diff = model.clf.weight[0] - model.clf.weight[1]
        dst_inds = self.aux_alg_inds(src_vec, clf_diff)
        with torch.no_grad():
            model.clf.weight.copy_(model.clf.weight[:, dst_inds])
        clf_diff = model.clf.weight[0] - model.clf.weight[1]
        print((clf_diff @ src_vec).item())

    def _replace_clf(self, task_id):
        print("Replace the classifier:", end=' ')
        torch.manual_seed(self.args.classifier_seed + task_id)
        self.model = self._Model(self.shared_weights).to(self.device)
        with torch.no_grad():
            self.permute_clf(self.src_vec, self.model)
        print("Complete.")


class ProcessRepository:
    baseline = RawProcess
    EWC = EWC_Process
    MAS = MAS_Process
    SCP = SCP_Process
    SI = SI_Process
    RWalk = Rwalk_Process
    Neuron = Neuron_Process
    NeuGrad = NeuGradProcess
    GradCAM = GradCAMProcess
    NeuGradVar = NeuGradVarProcess
    itpEWC = itpEWC_Process
    pseudoNPC = pseudoNPC_Process
    freezeClf = FreezeClfProcess
    reorderFreezeClf = ReorderFreezeClfProcess


class AgentController:
    def __init__(self, experiment, args):
        self.experiment = experiment
        self.args = args
        self._load_data_seq(args)
        self._load_prototype(args)

    def register_approach(self, approach):
        if not getattr(self, approach, None):
            newProcess = getattr(ProcessRepository, approach)
            newAgent = newProcess(self.args, self._Model)
            setattr(self, approach, newAgent)

    def run(self, approach):
        self.register_approach(approach)
        cur_agent = getattr(self, approach)
        print("Approach {} start running.".format(approach))
        cur_agent.run(self.data_seq)
        print("Approach {} complete.".format(approach))

    def run_multi(self, approaches):
        for approach in approaches:
            self.run(approach)

    def _load_data_seq(self, args):
        data_path = args.data_path
        experiment = args.experiment
        data_name = args.dataset
        batch_size = args.batch_size
        tasks = args.tasks
        print("Loading the {} data_seq:".format(experiment), end=' ')

        if experiment == "Permuted5_MNIST":
            dataset = utils.Permuted_Dataset(shuffle_size=14, tasks=tasks)
        elif "Split" in experiment:
            slices_path = args.slices_path
            dataset = utils.Split_Dataset(slices_path, tasks=tasks)

        train_seq = dataset.load_data_seq(data_path, data_name, batch_size, True)
        test_seq = dataset.load_data_seq(data_path, data_name, batch_size, False)
        reg_spec_seq = dataset.load_data_seq(data_path, data_name, 1, True)
        self.data_seq = {"train":train_seq, "test":test_seq, "reg_spec":reg_spec_seq}
        print("Complete.")

    def _load_prototype(self, args):
        try:
            self._Model = getattr(custom_models, args.model)
        except:
            raise FileNotFoundError
