import numpy as np
import matplotlib.pyplot as plt


def plot_multi_tasks_acc(tasks, all_test_acc, experiment):
    x = np.arange(1, tasks+1, 1)
    mark = ['o', '^', 's', '*', 'D',
            'v', 'p', '8', '<', '>']
    min_acc = 1
    for name, one_test_acc in all_test_acc.items():
        for idx, one_task in one_test_acc.items():
            temp = np.array(one_task)
            acc = temp[:, 1]
            if min_acc > np.min(acc):
                min_acc = np.min(acc)
    offset = (1 - min_acc) * 0.04
    for name, one_test_acc in all_test_acc.items():
        for idx, one_task in one_test_acc.items():
            temp = np.array(one_task)
            acc = temp[:, 1]
            plt.plot(x[idx:], acc, marker=mark[idx], markersize=8)
        # plt.axis([0.9, tasks+0.10, min_acc-offset, 1+offset])
        # Split5_MNIST
        # plt.axis([0.9, tasks+0.10, 0.835, 1.004])
        # Split5_FashionMNIST
        plt.axis([0.9, tasks+0.10, 0.500, 1.02])
        plt.grid()
        plt.title(name)
        # plt.savefig("result/"+name+".png", dpi=720)
        plt.show()

def cal_global_acc(tasks, all_test_acc):
    global_acc = dict()
    one_task_mean_acc = dict()
    for alg_name, seq in all_test_acc.items():
        s = 0
        temp = list()
        for idx, acc in seq.items():
            one_acc = 0
            for data in acc:
                s += data[1]
                one_acc += data[1]
            temp.append(one_acc / (tasks-idx))
        total_num = tasks * (tasks+1) / 2
        global_acc[alg_name] = s / (total_num)
        one_task_mean_acc[alg_name] = temp
    return global_acc, one_task_mean_acc

def cal_final_acc(tasks, all_test_acc):
    final_acc = dict()
    for alg_name, seq in all_test_acc.items():
        s = 0
        for idx, acc in seq.items():
            data = acc[-1]
            s += data[1]
        final_acc[alg_name] = s / tasks
    return final_acc

def cal_forget_seq(tasks, all_test_acc):
    forget_seq = dict()
    for key, multi_acc in all_test_acc.items():
        mean_forget = 0
        for i in range(4):
            acc = [acc for (loss, acc) in multi_acc[i]]
            mean_forget += (max(acc) - acc[-1])
        mean_forget /= (tasks-1)
        forget_seq[key] = mean_forget
    return forget_seq

def cal_forget_seq_2(tasks, all_test_acc):
    forget_seq = dict()
    for key, multi_acc in all_test_acc.items():
        mean_forget = 0
        for i in range(4):
            acc = [acc for (loss, acc) in multi_acc[i]]
            mean_forget += (acc[0] - min(acc))
        mean_forget /= (tasks-1)
        forget_seq[key] = mean_forget
    return forget_seq

def cal_learn_seq(tasks, all_test_acc):
    learn_seq = dict()
    base_acc = list()
    for i in range(tasks-1):
        loss, acc = all_test_acc["baseline"][i][0]
        base_acc.append(acc)
    for key, multi_acc in all_test_acc.items():
        mean_learn = 0
        for i in range(4):
            acc = [acc for (loss, acc) in multi_acc[i]]
            mean_learn += (base_acc[i] - max(acc))
        mean_learn /= (tasks-1)
        learn_seq[key] = mean_learn
    return learn_seq

def plot_forget_seq(tasks, all_test_acc):
    forget_plot = dict()
    for key, multi_acc in all_test_acc.items():
        forget = list()
        for i in range(5):
            acc = [acc for (loss, acc) in multi_acc[i]]
            forget.append((max(acc) - acc[-1]))
        forget_plot[key] = forget

    x = np.arange(1, tasks+1, 1)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=720)
    marks = ['o', 'v', '^', '<', '>',
             '8', 's', 'p', '*', 'h']
    ax.set_xticks(x)
    for idx, (alg_name, forget) in enumerate(forget_plot.items()):
        y = forget
        plt.plot(x, y, label=alg_name, marker=marks[idx])
        plt.legend()
    plt.xlabel("Task")
    plt.ylabel("Forgetting")
    plt.grid()
    plt.show()

def calc_baseline_ipt(model, shared_weights, all_models,
                      calc_ipt, train_seq, all_seq_ipts, tasks):
    from copy import deepcopy
    import torch

    ipt_weights = {}
    for name, param in model.named_parameters():
        block_name = name.split('.')[0]
        if block_name not in shared_weights:
            continue
        ipt_weights[name] = torch.zeros_like(param)
    zero_ipts = deepcopy(ipt_weights)

    seq_ipts = list()
    for i in range(tasks):
        train_dataloader = train_seq[i]
        model = deepcopy(all_models["Origin"])[i]
        # 计算EWC
        calc_ipt(model, ipt_weights, train_dataloader)
        # 保存重要性权重
        seq_ipts.append(deepcopy(ipt_weights))
        ipt_weights = deepcopy(zero_ipts)

    all_seq_ipts["Origin"] = seq_ipts

def calc_random_ipt(prototype, device, shared_weights,
                    model_path, model, train_seq, tasks,
                    calc_ipt, SEED, all_seq_ipts):
    import os
    import torch
    from copy import deepcopy

    init_model = prototype(device, shared_weights)
    # 载入初始化模型
    if os.path.exists(model_path):
        init_model.load_state_dict(torch.load(model_path))
        print("Load model from {}.".format(model_path))

    ipt_weights = {}
    for name, param in model.named_parameters():
        block_name = name.split('.')[0]
        if block_name not in shared_weights:
            continue
        ipt_weights[name] = torch.zeros_like(param)
    zero_ipts = deepcopy(ipt_weights)

    seq_ipts = list()
    for i in range(tasks):
        train_dataloader = train_seq[i]
        # 计算EWC
        calc_ipt(init_model, ipt_weights, train_dataloader)
        # 保存重要性权重
        seq_ipts.append(deepcopy(ipt_weights))
        ipt_weights = deepcopy(zero_ipts)
        torch.manual_seed(i+SEED)
        init_model = prototype(device, shared_weights)

    all_seq_ipts["Random"] = seq_ipts

def peel_fisher_matrix(all_seq_ipts, model, shared_weights):
    from copy import deepcopy
    import torch

    temp_seq_ipts = deepcopy(all_seq_ipts["EWC"])
    zero_ipts = {}
    for name, param in model.named_parameters():
        block_name = name.split('.')[0]
        if block_name not in shared_weights:
            continue
        zero_ipts[name] = torch.zeros_like(param)

    seq_ipts = [temp_seq_ipts[0]]
    for i in range(1, 5):
        temp_ipt_weights = temp_seq_ipts[i]
        new_ipt_weights = deepcopy(zero_ipts)
        for name in zero_ipts.keys():
            new_ipt_weights[name] = temp_ipt_weights[name] - seq_ipts[i-1][name]
        # 保存重要性权重
        seq_ipts.append(deepcopy(new_ipt_weights))

    all_seq_ipts["src_EWC"] = seq_ipts

def plot_ipt_scores(all_seq_ipts, alg_name, param_name):
    from copy import deepcopy

    ewc_ipts = list()
    row = 0
    out_channels = all_seq_ipts[alg_name][0][param_name].shape[0]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=720)
    for ipts in deepcopy(all_seq_ipts[alg_name]):
        obs_data = ipts[param_name].cpu().numpy()
        ewc_ipts.append(obs_data[row].reshape(1,-1))
    seq_ewc_ipt = np.concatenate(ewc_ipts)
    print(seq_ewc_ipt.shape)

    # 设置标签
    labels = ["Task"+str(i+1) for i in range(5)]
    # ax.set_xticks(np.arange(150))
    ax.set_yticks(np.arange(len(labels)))
    # ax.set_xticklabels(labels=labels)
    ax.set_yticklabels(labels=labels)
    # fig.set_figwidth(40)
    # fig.set_figheight(1)
    # ax.imshow(seq_ewc_ipt[0].reshape(1,-1))
    pos = ax.imshow(seq_ewc_ipt, cmap="turbo")
    plt.colorbar(pos, ax=ax, shrink=0.5)
    ax.set_aspect(20)
    plt.xlabel("Important Scores")
    ax.xaxis.set_major_locator(plt.NullLocator())
    # plt.grid(axis='y')
    # plt.grid()
    hlines = np.arange(0.5, 5, 1)
    for i in range(5):
        ax.axhline(hlines[i], color="white", linestyle='--', alpha=0.2)
    vlines = np.arange(25, 150, 25)
    for i in range(5):
        ax.axvline(vlines[i], color="white", linestyle='--', alpha=0.2)
    # plt.savefig("sparse_ipt.png", dpi=720)
    plt.show()

def plot_ipt_scores_2(all_seq_ipts, alg_name, param_name):
    from torch import nn
    from copy import deepcopy

    softmax = nn.Softmax(dim=0)
    ewc_ipts = list()
    instance = all_seq_ipts[alg_name][0][param_name]
    out_channels, in_channels = instance.shape[:2]
    if instance.ndim > 2:
        size = instance.shape[2] * instance.shape[3]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=720)
    for ipts in deepcopy(all_seq_ipts[alg_name]):
        obs_data = ipts[param_name].cpu().numpy()
        if instance.ndim > 2:
            ewc_ipts.append(obs_data.reshape(out_channels*in_channels,-1).T)
        else:
            ewc_ipts.append(obs_data)
    seq_ewc_ipt = np.concatenate(ewc_ipts)
    print(seq_ewc_ipt.shape)

    # 设置标签
    labels = ["Task"+str(i+1) for i in range(5)]
    if instance.ndim > 2:
        ax.set_xticks(np.arange(0, in_channels*out_channels, in_channels))
        ax.set_yticks(np.arange(0, len(labels)*size, size))
    else:
        ax.set_xticks(np.arange(0, in_channels, 1))
        ax.set_yticks(np.arange(0, len(labels)*out_channels, out_channels))
    # ax.set_xticklabels(labels=labels)
    ax.set_yticklabels(labels=labels)

    pos = ax.imshow(seq_ewc_ipt, cmap="turbo")
    # pos = ax.imshow(np.log(seq_ewc_ipt), cmap="turbo")
    ratio = seq_ewc_ipt.shape[1] / seq_ewc_ipt.shape[0]
    plt.colorbar(pos, ax=ax, shrink=ratio)
    ax.set_aspect(ratio)
    plt.xlabel("Important Scores")
    ax.xaxis.set_major_locator(plt.NullLocator())

    if instance.ndim > 2:
        hlines = np.arange(0, len(labels)*size, size)-0.5
    else:
        hlines = np.arange(0, len(labels)*out_channels, out_channels)-0.5
    for i in range(5):
        ax.axhline(hlines[i], color="white", linestyle='--', alpha=0.2, linewidth=0.8)

    if instance.ndim > 2:
        vlines = np.arange(0, in_channels*out_channels, in_channels)-0.5
        for i in range(out_channels):
            ax.axvline(vlines[i], color="white", linestyle='--', alpha=0.2, linewidth=0.8)

    # plt.savefig("Origin_FIM.png", dpi=720)
    plt.show()

def plot_parameter_distance(all_models, alg_name, param_name, init_model):
    import os
    import torch
    from copy import deepcopy

    init_params = init_model.state_dict()[param_name].cpu().numpy()
    row = 0
    dst_param = init_params[row].reshape(1,-1)

    model_params = list()
    out_channels = all_models[alg_name][0].state_dict()[param_name].shape[0]
    fig, ax = plt.subplots(figsize=(5, 5), dpi=720)
    for m in deepcopy(all_models[alg_name]):
        obs_data = m.state_dict()[param_name].cpu().numpy()
        model_params.append(obs_data[row].reshape(1,-1))
    seq_models = np.concatenate(model_params)
    param_diffs = np.zeros_like(seq_models)
    param_diffs[0] = np.log10((dst_param - model_params[0]) ** 2)
    for i in range(1, 5):
        param_diffs[i] = np.log10((model_params[i] - model_params[i-1]) ** 2)
    print(param_diffs.shape)

    # 设置标签
    labels = ["Task"+str(i+1) for i in range(5)]
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels=labels)

    pos = ax.imshow(param_diffs, cmap="turbo")
    plt.colorbar(pos, ax=ax, shrink=0.5)
    ax.set_aspect(20)
    plt.xlabel("Parameter Distance")
    ax.xaxis.set_major_locator(plt.NullLocator())

    hlines = np.arange(0.5, 5, 1)
    for i in range(5):
        ax.axhline(hlines[i], color="white", linestyle='--', alpha=0.2, linewidth=0.8)
    vlines = np.arange(25, 150, 25)
    for i in range(5):
        ax.axvline(vlines[i], color="white", linestyle='--', alpha=0.2, linewidth=0.8)
    # plt.savefig("param_distance.png", dpi=720)
    plt.show()

def plot_multi_tasks_bar(tasks, all_test_acc):
    x = np.arange(1, tasks, 1)
    for name, one_test_acc in all_test_acc.items():
        offset = -0.5
        for idx, one_task in one_test_acc.items():
            temp = np.array(one_task)
            acc = temp[:, 1] * 100
            plt.bar(x[idx:]+offset, acc, width=0.2, align='edge')
            offset += 0.2
        plt.title(name)
        plt.grid()
        plt.show()

def plot_contrast_experiments(tasks, all_test_acc):
    x = np.arange(1, tasks+1, 1)
    src_palette = ["#00CDCD", "#FF6A6A",
                   "#6A5ACD", "#FFD700",
                   "#7F7F7F", "#0214F3"]
    palette = iter(src_palette)

    algs = ["Origin", "EWC", "MAS", "Neuron", "SCM", "Uniform"]
    alg_iter = iter(algs)
    fig = plt.figure(figsize=(4, 2), dpi=720)
    axes = fig.add_axes([0, 0, 1, 1])
    major_x = np.linspace(0, 5, 6)
    major_y = np.linspace(0, 1, 6)
    minor_y = np.linspace(0, 1, 21)

    axes.set_xticks(major_x)
    axes.set_yticks(major_y)
    axes.set_yticks(minor_y, minor=True)
    axes.grid(which="major", alpha=0.6)
    axes.grid(which="minor", alpha=0.3)

    offset = -0.4
    space = 0.8 / len(algs)
    for name, one_test_acc in all_test_acc.items():
        cur_color = next(palette)
        cur_alg = next(alg_iter)
        for idx, one_task in one_test_acc.items():
            temp = np.array(one_task)
            x_coor = x[idx] + offset
            # 下面的代码只绘制最终结果
            ################################################
            # final_acc = temp[-1, 1]
            # init_acc = temp[0, 1]
            # plt.bar(x_coor, final_acc, width=0.2,
            #         align='edge', color=cur_color)
            # plt.bar(x_coor, init_acc-final_acc, width=0.2,
            #         align='edge', color="black", alpha=1,
            #         bottom=final_acc, linestyle="--", fill=False)
            ################################################
            
            # 下面的代码绘制平均准确率
            ################################################
            acc = temp[:, 1]
            mean_acc = np.mean(acc)
            bar = plt.bar(x_coor, mean_acc, width=space,
                        align='edge', color=cur_color)
            ################################################
        offset += space
    plt.title("Contrast Experiments")
    # plt.savefig("Contrast Experiments.png", dpi=720)
    plt.show()

def plot_parameter_hist(N, model, layer, print_data=False):
    param = model.state_dict()[layer].cpu().numpy()
    if param.ndim > 1:
        for idx, p in enumerate(param):
            plt.title(layer+"(out_channel: {})".format(idx))
            count, bins = np.histogram(p, bins=N)
            count = np.array(count) / len(p)
            # plt.stairs(count, bins)
            med_bins = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
            width = (bins[-1]-bins[0]) / (N+1)
            plt.bar(med_bins, count, width=width)
            plt.grid()
            # plt.plot(med_bins, count, "--", c='r')
            plt.show()
            if print_data:
                print("bins:{}, count:{}".format(med_bins, count))
    else:
        p = param
        plt.title(layer)
        count, bins = np.histogram(p, bins=N)
        count = np.array(count) / len(p)
        # plt.stairs(count, bins)
        med_bins = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
        width = (bins[-1]-bins[0]) / (N+1)
        plt.bar(med_bins, count, width=width)
        plt.grid()
        # plt.plot(med_bins, count, "--", c='r')
        plt.show()
        if print_data:
            print("bins:{}, count:{}".format(med_bins, count))

def plot_multi_tasks_acc_2(tasks, all_test_acc, experiment, order):
    x = np.arange(1, tasks+1, 1)
    mark = ['o', '^', 's', '*', 'D',
            'v', 'p', '8', '<', '>']
    for name, one_test_acc in all_test_acc.items():
        for idx, one_task in one_test_acc.items():
            temp = np.array(one_task)
            acc = temp[:, 1]
            plt.plot(x[idx:], acc, marker=mark[idx],
                     markersize=8, label=f"Task{order[idx]+1}")    
            if "CIFAR10" in experiment:
                plt.axis([0.95, tasks+0.10, 0.33, 1.02])
            # elif "MNIST" in experiment:
            #     plt.axis([0.95, tasks+0.10, 0.94, 1.002])
            # plt.axis([0.95, tasks+0.10, 0.78, 1.01])
        plt.grid()
        plt.title(name)
        # plt.savefig("result/"+name+".png", dpi=720)
        plt.legend()
        # plt.savefig("all_test_acc-14230.png", dpi=720)
        plt.show()

def plot_hist(data, n, title):
    count, bins = np.histogram(data, bins=n)
    count = np.array(count) / len(data)
    # plt.stairs(count, bins)
    med_bins = [(bins[i]+bins[i+1])/2 for i in range(len(bins)-1)]
    width = (bins[-1]-bins[0]) / (n+1)
    plt.title(title)
    plt.bar(med_bins, count, width=width)
    plt.yscale('log')
    plt.grid()
    # plt.plot(med_bins, count, "--", c='r')
    plt.show()
