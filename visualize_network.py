# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 16:41:34 2023

@author: Admin
"""

import sys, os

import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QCheckBox, QSpinBox, QDoubleSpinBox,
                             QFrame, QFileDialog, QMessageBox, QHBoxLayout,
                             QVBoxLayout, QAction, QTableView, QSizePolicy,
                             QComboBox)
from PyQt5.QtGui import QStandardItem, QStandardItemModel, QPixmap, QImage

from custom_models import MLP2, LeNet2
from PyQt5.QtCore import Qt
from copy import deepcopy
from utils import RawTrain, RegularTrain


class VisualizeNetwork(QMainWindow):

    def __init__(self, agent, data_seq):
        super().__init__()
        self.model = deepcopy(agent.model)
        self.data_seq = data_seq
        self.optimizer = agent._get_optim(self.model.parameters())
        self.loss_fn = agent.loss_fn
        self.tasks = agent.tasks
        self.param_windows = dict()
        self.step_commands = dict()
        self.networkProcess = Process(self.model, self.data_seq,
                                      self.optimizer, self.loss_fn)
        self.taskID = None
        self.relativeImageID = None
        self.neuronID = None
        self.initUI()

    def initUI(self):
        """Initialize the window and display its contents to the screen."""
        self.img_width = 200
        self.img_height = 200
        self.setupWindow()
        self.show()

    def setupWindow(self):
        self.main_h_box = QHBoxLayout()
        self.ImageDisplayBox()
        self.networkView()
        self.updateParams()
        self.neuronsDisplayBox()

        container = QWidget()
        container.setLayout(self.main_h_box)
        self.setCentralWidget(container)

        # side_panel_frame = QFrame()
        # side_panel_frame.setMinimumWidth(200)
        # side_panel_frame.setFrameStyle(QFrame.WinPanel)
        # side_panel_frame.setLayout(side_panel_v_box)

        # side_panel_v_box = QVBoxLayout()
        # side_panel_v_box.setAlignment(Qt.AlignTop)
        # side_panel_v_box.addWidget(contrast_label)

    def networkView(self):
        networkBox = QVBoxLayout()
        for name, param in self.model.named_parameters():
            synapse_button = QPushButton(name)
            # self.synapse_button.setGeometry(200, 150, 100, 40)
            # synapse_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
            self.param_windows[name] = SubWindows(name, param, self.writeDatatoTable)
            synapse_button.clicked.connect(self.param_windows[name].show)
            networkBox.addWidget(synapse_button)

        network_panel = QWidget()
        network_panel.setLayout(networkBox)
        self.main_h_box.addWidget(network_panel)

    def updateParams(self):
        update_box = QVBoxLayout()
        commands = ["step_one_batch",
                    "step_one_epoch",
                    "step_one_task"]

        for command in commands:
            command_type = command.split('_')[-1]
            update_param_button = QPushButton(command_type + ' +1')
            self.step_commands[command] = getattr(self.networkProcess, command)
            update_param_button.clicked.connect(self.step_commands[command])
            update_box.addWidget(update_param_button)
        
        update_param_panel = QWidget()
        update_param_panel.setLayout(update_box)
        self.main_h_box.addWidget(update_param_panel)

    def ImageDisplayBox(self):
        displayBox = QVBoxLayout()
        self.task_sel_box = QComboBox()
        self.task_sel_box.addItem("[Select the task ID]")
        for task_id in range(self.tasks):
            self.task_sel_box.addItem(str(task_id+1))
        self.task_sel_box.activated.connect(self.load_image_indices)
        displayBox.addWidget(self.task_sel_box)

        self.image_sel_box = QComboBox()
        displayBox.addWidget(self.image_sel_box)

        self.image_label = QLabel()
        displayBox.addWidget(self.image_label)

        display_panel = QWidget()
        display_panel.setLayout(displayBox)
        self.main_h_box.addWidget(display_panel)

    def load_image_indices(self, index):
        if index == 0:
            print("Cannot select the current option")
            return False

        train_seq = self.data_seq["train"]
        task_id = int(self.task_sel_box.itemText(index))-1
        self.taskID = task_id
        train_dataloader = train_seq[task_id]
        self.image_sel_box.clear()
        self.image_sel_box.addItem("[Select the image ID]")
        for i in train_dataloader.sampler:
            self.image_sel_box.addItem(str(i))

        self.image_sel_box.activated.connect(self.displayImage)

    def displayImage(self, index):
        import torch
        if index == 0:
            print("Cannot select the current option.")
            return False

        def np_to_qimage(image):
            height, width, channels = image.shape
            bytes_per_line = width * channels
            converted_Qt_image = QImage(bytes(image), width, height,
                                        bytes_per_line, QImage.Format_RGB888)
            return converted_Qt_image

        image_id = int(self.image_sel_box.itemText(index))
        self.relativeImageID = int(index)-1
        image = deepcopy(self.data_seq["train"][0].dataset.data[image_id])
        if torch.is_tensor(image):
            image = image.numpy()
        if image.ndim < 3:
            image = np.expand_dims(image, axis=2)
            image = np.tile(image,(1,1,3))
        converted_image = np_to_qimage(image)

        self.image_label.setPixmap(QPixmap.fromImage(converted_image).scaled(
            self.img_width, self.img_height, Qt.KeepAspectRatioByExpanding))

    def neuronsDisplayBox(self):
        displayBox = QVBoxLayout()
        self.neuron_sel_box = QComboBox()
        self.neuron_sel_box.addItem("[Select the neuron - the output of {*}]")
        all_modules = getattr(self.networkProcess, "all_modules")
        for module in all_modules:
            self.neuron_sel_box.addItem(module)
        displayBox.addWidget(self.neuron_sel_box)

        neuronView = QTableView()
        self.qtNeuronModel = QStandardItemModel()
        neuronView.setModel(self.qtNeuronModel)
        displayBox.addWidget(neuronView)

        self.neuron_sel_box.activated.connect(self.update_neuronID)

        forPropButton = QPushButton("Forward Propagation")
        displayBox.addWidget(forPropButton)
        forPropButton.clicked.connect(self.update_neuron)
        
        display_panel = QWidget()
        display_panel.setLayout(displayBox)
        self.main_h_box.addWidget(display_panel)

    def update_neuronID(self, index):
        if index == 0:
            print("Cannot select the current option.")
            return False

        self.neuronID = self.neuron_sel_box.itemText(index)

    def update_neuron(self):
        self.networkProcess.forward(self.taskID, self.relativeImageID)
        if self.neuronID is None:
            print("neuron ID is None.")
            return False
        neurons = self.networkProcess.rec_neurons[self.neuronID]
        self.writeDatatoTable(neurons, self.qtNeuronModel)

    def writeDatatoTable(self, data, qtDataModel):
        out_channels = data.shape[0]
        flatten_param = data.reshape(out_channels, -1)
        for row, one_bank_param in enumerate(flatten_param):
            for col, param in enumerate(one_bank_param):
                item = QStandardItem(str(param.item()))
                qtDataModel.setItem(row, col, item)
        qtDataModel.itemChanged.connect(self.test)

    def test(self):
        print("hello")


class SubWindows(QWidget):
    def __init__(self, name, param, writeDatatoTable):
        super().__init__()
        self.name = name
        self.param = param
        self.writeDatatoTable = writeDatatoTable
        self.initUI()

    def initUI(self):
        synapses_box = self.make_synapses_box(self.param)
        self.setLayout(synapses_box)

    def make_synapses_box(self, param_banks):
        synapses_box = QVBoxLayout()
        resultView = QTableView()
        qtDataModel = QStandardItemModel()

        self.writeDatatoTable(param_banks, qtDataModel)
        resultView.setModel(qtDataModel)
        synapses_box.addWidget(resultView)
        return synapses_box


class Process:
    def __init__(self, model, data_seq, optimizer,
                 loss_fn, reg_groups=None):
        self.model = model
        self.data_seq = data_seq
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.reg_groups = reg_groups
        self.cur_batch = 0
        self.cur_epoch = 0
        self.cur_task_id = 0
        self.epochs = 10
        self.tasks = 5
        if reg_groups is None:
            self.train = RawTrain().base_train
        else:
            self.train = RegularTrain().base_train
        self.make_iterator()
        self.all_tasks_complete = False
        self.handles = list()
        self.all_modules = list()
        self.get_moduleID()
        self.rec_neurons = dict()

    def make_iterator(self):
        train_seq = self.data_seq["train"]
        self.train_iter = iter(train_seq[self.cur_task_id])

    def update_status(self):
        self.cur_epoch += 1
        print("current epoch: {}".format(self.cur_epoch))
        if self.cur_epoch >= self.epochs:
            self.cur_epoch = 0
            self.cur_task_id += 1
            print("Task-{} Complete".format(self.cur_task_id))
            if self.cur_task_id >= self.tasks:
                self.all_tasks_complete = True

    def reload_data(self):
        # restart the new step
        self.make_iterator()
        self.cur_batch = 0

    def run_one_batch(self):
        try:
            X, y = next(self.train_iter)
            self.cur_batch += 1
        except:
            return False
        self.train(self.model, X, y, self.optimizer,
                   self.loss_fn, self.reg_groups, self.cur_batch)
        return True
            
    def step_one_batch(self):
        status = self.run_one_batch()
        if not status:
            self.update_status()
            self.reload_data()

    def step_one_epoch(self):
        while self.run_one_batch():
            pass
        self.update_status()
        self.reload_data()

    def step_one_task(self):
        for i in range(self.cur_epoch, self.epochs):
            while self.run_one_batch():
                pass
            self.make_iterator()
            self.cur_batch = 0
            print("current epoch: {}".format(self.cur_epoch))

        self.cur_epoch = 0
        self.cur_task_id += 1
        print("Task-{} Complete".format(self.cur_task_id))
        if self.cur_task_id >= self.tasks:
            self.all_tasks_complete = True
        else:
            self.make_iterator()

    def forward(self, taskID, relativeImageID):
        moduleIDs = iter(self.all_modules)
        def fhook(module, f_in, f_out):
            self.rec_neurons[next(moduleIDs)] = f_out

        for name, module in self.model.named_modules():
            if name not in self.all_modules:
                continue
            self.handles.append(module.register_forward_hook(fhook))

        reg_spec_seq = self.data_seq["reg_spec"]
        X, y = list(reg_spec_seq[taskID])[relativeImageID]
        self.model(X)

        for handle in self.handles:
            handle.remove()

    def get_moduleID(self):
        from torch import nn

        for name, module in self.model.named_modules():
            # The first module is the network.
            # Its name is '' and we skip it.
            if name == '' or isinstance(module, nn.Sequential):
                continue
            self.all_modules.append(name)


# class SubWindows(QWidget):
#      def __init__(self, model):
#          super(SubWindows, self).__init__()
#          self.resize(400, 300)

#          # Label
#          self.label = QLabel(self)
#          self.label.setGeometry(0, 0, 400, 300)
#          self.label.setText('Sub Window')
#          self.label.setAlignment(Qt.AlignCenter)
#          self.label.setStyleSheet('font-size:40px')

if __name__ == '__main__':
    import yaml
    from agent import AgentController
    
    
    if __name__ == "__main__":
        with open ("config.yaml", "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
    
        class CustomArgs:
            def __init__(self, dataset, config):
                for name, value in config[dataset].items():
                    setattr(self, name, value)

    # experiment = "Split5_MNIST"
    experiment = "Split5_CIFAR10"
    args = CustomArgs(experiment, config)
    # print(args.__dict__)

    admin = AgentController(experiment, args)
    approach = "baseline"
    admin.register_approach(approach)
    agent = getattr(admin, approach)
    agent.reset()
    model = agent.model
    data_seq = admin.data_seq
    optimizer = agent._get_optim(model.parameters())
    loss_fn = agent.loss_fn

    app = QApplication(sys.argv)
    window = VisualizeNetwork(agent, data_seq)
    sys.exit(app.exec_())
