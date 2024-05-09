from collections import OrderedDict
from torch import nn
import torch.nn.functional as F


########################################################
######################## LeNet2 ########################
# split_CIFAR10/100 - simple
class LeNet2(nn.Module):
    def __init__(self, shared_weights=dict()):
        super(LeNet2, self).__init__()
        self.backbone = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 6, 5)),
            ("relu1", nn.ReLU()),
            ("max_pool1", nn.MaxPool2d((2, 2))),
            ("conv2", nn.Conv2d(6, 16, 5)),
            ("relu2", nn.ReLU()),
            ("max_pool2", nn.MaxPool2d(2)),
        ]))
        self.bridge = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(16*5*5, 120)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(120, 84)),
            ("relu2", nn.ReLU())
        ]))
        if "backbone" in shared_weights:
            self.backbone = shared_weights["backbone"]
        if "bridge" in shared_weights:
            self.bridge = shared_weights["bridge"]
        self.clf = nn.Linear(84, 2)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = self.bridge(x)
        x = self.clf(x)
        return x

########################################################
######################### MLP10 ########################
# permuted_MNIST - normal
class MLP10(nn.Module):
    def __init__(self, shared_weights=dict()):
        super(MLP10, self).__init__()
        self.flatten = nn.Flatten()
        self.backbone = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(28*28, 2000)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(2000, 2000)),
            ("relu2", nn.ReLU()),
        ]))
        if "backbone" in shared_weights:
            self.backbone = shared_weights["backbone"]
        self.clf = nn.Linear(2000, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.backbone(x)
        x = self.clf(x)
        return x

########################################################
######################### MLP2 #########################
# split_MNIST - normal
class MLP2(nn.Module):
    def __init__(self, shared_weights=dict()):
        super(MLP2, self).__init__()
        self.flatten = nn.Flatten()
        self.backbone = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(28*28, 256)),
            ("relu1", nn.ReLU()),
            ("fc2", nn.Linear(256, 256)),
            ("relu2", nn.ReLU()),
        ]))
        if "backbone" in shared_weights:
            self.backbone = shared_weights["backbone"]
        self.clf = nn.Linear(256, 2)

    def forward(self, x):
        x = self.flatten(x)
        x = self.backbone(x)
        x = self.clf(x)
        return x


########################################################
###################### CIFAR_CNN #######################
# split_CIFAR10/100 - normal
class CIFAR_CNN(nn.Module):
    def __init__(self, tasks):
        super(CIFAR_CNN, self).__init__()
        self.backbone = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(3, 32, 3)),
            ("relu1", nn.ReLU()),
            ("conv2", nn.Conv2d(32, 32, 3)),
            ("relu2", nn.ReLU()),
            ("max_pool1", nn.MaxPool2d((2, 2))),
            ("conv3", nn.Conv2d(32, 64, 3)), 
            ("relu3", nn.ReLU()),
            ("conv4", nn.Conv2d(64, 64, 3)),
            ("relu4", nn.ReLU()),
            ("max_pool2", nn.MaxPool2d(2)),
        ]))
        self.bridge = nn.Linear(64*5*5, 512)
        self.clfs = nn.ModuleList()
        for i in range(tasks):
            clf = nn.Linear(512, 100 // tasks)
            self.clfs.append(clf)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.bridge(x))
        preds = list()
        for clf in self.clfs:
            preds.append(clf(x))
        return preds