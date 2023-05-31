import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets

from stream_controller import StreamController


class OptionsWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(OptionsWindow, self).__init__()

        self.setWindowTitle("AVS")
        self.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.setCentralWidget(self.centralwidget)

        self.left_bpm_label = QtWidgets.QLabel("LBPM: -")
        self.right_bpm_label = QtWidgets.QLabel("RBPM: -")

        self.verticalLayout.addWidget(self.left_bpm_label)
        self.verticalLayout.addWidget(self.right_bpm_label)