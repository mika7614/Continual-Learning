# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 11:20:24 2023

@author: Admin
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget,\
                            QDesktopWidget, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt


class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Visualize the network'
        self.width = 640
        self.height = 480
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.align_center()

    def align_center(self):
        screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.left = int((screen.width() - size.width()) / 2)
        self.top = int((screen.height() - size.height()) / 2)
        self.setGeometry(self.left, self.top, self.width, self.height)

    def np_to_qimage(self, image):
        height, width, channels = image.shape
        bytes_per_line = width * channels
        converted_Qt_image = QImage(bytes(image), width, height,
                                    bytes_per_line, QImage.Format_RGB888)
        return converted_Qt_image

    def main(self, image):
        label = QLabel(self)
        converted_image = self.np_to_qimage(image)
        label.setPixmap(QPixmap.fromImage(converted_image).scaled(
            self.width, self.height, Qt.KeepAspectRatioByExpanding))
        self.show()


if __name__ == '__main__':
    from utils import BaseDataset
    app = QApplication(sys.argv)
    ex = App()
    data = BaseDataset().load_dataset("data", "CIFAR10", train=True)
    image = data[0][0].numpy().transpose((1,2,0))
    image = (image * 255).astype(np.uint8)
    ex.main(image)
    sys.exit(app.exec_())
