import os
import pickle
from copy import deepcopy
import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose, Normalize
import numbers
from collections.abc import Sequence


class RawTrain:
    def train(self, model, dataloader,
              optimizer, loss_fn, reg_groups=None):
        size = len(dataloader.sampler)
        train_loss = 0
        for batch, (X, y) in enumerate(dataloader):
            loss = self.base_train(model, X, y, optimizer,
                                   loss_fn, reg_groups, batch)
            # Print the information
            if batch % 100 == 0:
                loss_value, current = loss.item(), batch * len(X)
                print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")
            train_loss += loss.item()
        return train_loss / len(dataloader)

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
        # self.freeze_layer(model)
        optimizer.step()
        # Online algorithms only
        self.update_reg_groups(model, reg_groups, batch)
        return loss

    def calc_reg(self, model, reg_groups, lr):
        return 0

    def update_reg_groups(self, model, reg_groups, batch=None):
        pass

    def freeze_layer(self, model):
        model.clf.weight.grad.zero_()
        model.clf.bias.grad.zero_()


class RegularTrain(RawTrain):
    def calc_reg(self, model, reg_groups, lr):
        prev_params = reg_groups["prev_params"] 
        coef = reg_groups["coef"]
        ipt_groups = deepcopy(reg_groups['ipt_groups'])

        for name, ipt in ipt_groups.items():
            ipt_groups[name] = torch.clamp(ipt, min=0, max=0.5/(coef*lr))

        s = 0
        for name, param in model.named_parameters():
            if name in ipt_groups:
                temp1 = (param - prev_params[name]).pow(2).reshape(1, -1)
                temp2 = ipt_groups[name].reshape(1, -1)
                s += coef * temp1 @ temp2.t()
        return s


def test(model, dataloader, loss_fn):
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
    return test_loss, acc

class BaseDataset:
    def load_dataset(self, data_path, data_name, train:bool=True):
        Dataset = getattr(datasets, data_name)
        # choice 1
#         transform = ToTensor()
        # choice 2
        transform = self.preprocess(data_name)

        data = Dataset(
            root=data_path,
            train=train,
            download=False,
            transform=transform,
        )
        return data

    def preprocess(self, data_name):
        if data_name == "MNIST":
            transform = Compose([
                ToTensor(),
#                 # train + test
#                 Normalize(
#                     mean=0.1309,
#                     std=0.3084
#                 )
                # train
                Normalize(
                    mean=0.1307,
                    std=0.3081
                )
            ])
        elif data_name == "CIFAR10":
            transform = Compose([
                ToTensor(),
#                 # train + test
#                 Normalize(
#                     mean=[0.4919, 0.4827, 0.4472],
#                     std=[0.2470, 0.2434, 0.2616]
#                 )
                # train
                Normalize(
                    mean=[0.4914, 0.4822, 0.4465],
                    std=[0.2470, 0.2435, 0.2616]
                )
            ])
        elif data_name == "CIFAR100":
            transform = Compose([
                ToTensor(),
#                 # train + test
#                 Normalize(
#                     mean=[0.5074, 0.4867, 0.4411],
#                     std=[0.2675, 0.2566, 0.2763]
#                 )
                # train
                Normalize(
                    mean=[0.5071, 0.4865, 0.4409],
                    std=[0.2673, 0.2564, 0.2762]
                )
            ])
        elif data_name == "FashionMNIST":
            transform = Compose([
                ToTensor(),
                # train
                Normalize(
                    mean=0.2860,
                    std=0.3530
                )
            ])
        return transform

    def load_data_seq(self, data_path, data_name, batch_size, train:bool=True):
        data = self.load_dataset(data_path, data_name, train)
        dataloader = DataLoader(data, batch_size=batch_size)
        data_seq = [dataloader]
        return data_seq

    def custom_sampler(self):
        class MySampler(Sampler):
            def __init__(self, data_source):
                self.data_source = data_source

            def __iter__(self):
                return iter(self.data_source)

            def __len__(self):
                return len(self.data_source)
        return MySampler


class Split_Dataset(BaseDataset):
    def __init__(self, slices_path, tasks):
        self.slices_path = slices_path
        self.tasks = tasks

    def split_dataset(self, data):
        if len(data.classes) % self.tasks == 0:
            subset_size = len(data.classes) // self.tasks
            for i in range(len(data.targets)):
                data.targets[i] = data.targets[i] % subset_size
        elif self.tasks == 4:
            for i in range(len(data.targets)):
                if data.targets[i] < 6:
                    data.targets[i] = data.targets[i] % 3
                else:
                    data.targets[i] = data.targets[i] % 2

        return data

    def load_data_seq(self, data_path, data_name, batch_size, train:bool=True):
        data = self.load_dataset(data_path, data_name, train)
        data = self.split_dataset(data)
        if train:
            data_path = self.slices_path["train"]
        else:
            data_path = self.slices_path["test"]
        with open(data_path, "rb") as f:
            data_inds = pickle.load(f)

        data_seq = list()
        for i in range(self.tasks):
            samples = np.array(data_inds[i], dtype=int)
            dataloader = DataLoader(data, batch_size=batch_size,
                                    sampler=samples)
            data_seq.append(dataloader)

        return data_seq

    def shuffle_indices(train_indices, test_indices, shuf_seed=None):
        np.random.seed(shuf_seed)
        np.random.shuffle(train_indices)
        np.random.seed(shuf_seed)
        np.random.shuffle(test_indices)


class Permuted_Dataset(BaseDataset):
    def __init__(self, shuffle_size, tasks):
        self.shuffle_size = shuffle_size
        self.tasks = tasks

    def permute_dataset(self, dataset, size: int, seed: int):
        if isinstance(size, numbers.Number):
            size = (int(size), int(size))

        if isinstance(size, Sequence) and len(size) == 1:
            size = (size[0], size[0])

        if len(size) != 2:
            raise ValueError("Please provide only "\
                            "two dimensions (h, w) for size.")

        # 以相同的顺序打乱中心图像块
        np_img = dataset.data[0].clone().detach().numpy()
        row_u = int((np_img.shape[0] - size[0]) / 2)
        row_d = int(row_u + size[0])
        col_l = int((np_img.shape[1] - size[1]) / 2)
        col_r = int(col_l + size[1])
        indices = np.meshgrid(np.arange(col_l, col_r),
                                np.arange(row_u, row_d))
        indices.reverse()

        # fix the random seed
        start_seed = 613
        for index in indices:
            np.random.seed(seed)
            np.random.shuffle(index.reshape((-1, 1)))
            seed += start_seed

        indices = tuple(indices)
        for img in dataset.data:
            np_img = img.clone().detach().numpy()
            np_img[row_u: row_d, col_l: col_r] = np_img[indices]
            img.copy_(torch.from_numpy(np_img))

        return dataset

    def load_data_seq(self, data_path, data_name, batch_size, train:bool=True):
        data = self.load_dataset(data_path, data_name, train)
        data_seq = list()

        for i in range(self.tasks):
            data = self.permute_dataset(data, self.shuffle_size, i)
            dataloader = DataLoader(data, batch_size=batch_size)
            data_seq.append(dataloader)

        return data_seq