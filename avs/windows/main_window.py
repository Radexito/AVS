import pyqtgraph as pg
from PySide6.QtCore import QThread
from PySide6.QtCore import Signal
from PySide6 import QtCore, QtWidgets
import matplotlib.pyplot as plt
from PySide6.QtGui import QAction

from avs.stream_controller import StreamController
from avs.windows.options import OptionsWindow


class UpdateLeftChannelThread(QThread):
    update_left_channel_signal = Signal()

    def run(self):
        while True:
            self.update_left_channel_signal.emit()
            self.msleep(1)


class UpdateRightChannelThread(QThread):
    update_right_channel_signal = Signal()

    def run(self):
        while True:
            self.update_right_channel_signal.emit()
            self.msleep(1)


class MainWindow(QtWidgets.QMainWindow):
    cmap = plt.get_cmap('coolwarm')

    def __init__(self, left_channel_index, right_channel_index):
        super(MainWindow, self).__init__()

        self.setWindowTitle("AVS")
        self.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(self)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)

        self.menubar = QtWidgets.QMenuBar(self)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.setMenuBar(self.menubar)

        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setTitle("File")

        self.actionOptions = QAction(self)
        self.actionOptions.setText("Options")
        self.actionOptions.triggered.connect(self.file_menu_options_clicked)

        self.menuFile.addAction(self.actionOptions)
        self.menubar.addAction(self.menuFile.menuAction())

        self.update_left_thread = UpdateLeftChannelThread()
        self.update_left_thread.update_left_channel_signal.connect(self.update_left_channel_ui)

        self.update_right_thread = UpdateRightChannelThread()
        self.update_right_thread.update_right_channel_signal.connect(self.update_right_channel_ui)

        self.setCentralWidget(self.centralwidget)

        self.left_channel_waveform = pg.PlotWidget(parent=self)
        self.left_channel_waveform.plotItem.hideAxis('bottom')
        self.left_channel_waveform.plotItem.hideAxis('left')

        self.right_channel_waveform = pg.PlotWidget(parent=self)
        self.right_channel_waveform.plotItem.hideAxis('bottom')
        self.right_channel_waveform.plotItem.hideAxis('left')

        self.verticalLayout.addWidget(self.left_channel_waveform)
        self.verticalLayout.addWidget(self.right_channel_waveform)

        self.left_channel_sc = StreamController()
        self.right_channel_sc = StreamController()

        self.verticalLayout.addWidget(self.left_channel_sc)
        self.verticalLayout.addWidget(self.right_channel_sc)

        self.left_pdataitem = self.left_channel_waveform.plot(self.left_channel_sc.data)
        self.right_pdataitem = self.right_channel_waveform.plot(self.right_channel_sc.data)

        self.left_channel_sc.setup_stream(left_channel_index)
        self.right_channel_sc.setup_stream(right_channel_index)

        self.left_bpm_label = QtWidgets.QLabel("LBPM: -")
        self.right_bpm_label = QtWidgets.QLabel("RBPM: -")

        self.verticalLayout.addWidget(self.left_bpm_label)
        self.verticalLayout.addWidget(self.right_bpm_label)

        self.update_left_thread.start()
        self.update_right_thread.start()

    def file_menu_options_clicked(self):
        self.options_window = OptionsWindow()
        self.options_window.show()

    def closeEvent(self, event):
        self.left_channel_sc.breakdown_stream()
        self.right_channel_sc.breakdown_stream()
        event.accept()

    def update_left_channel_ui(self):
        self.left_pdataitem.setData(self.left_channel_sc.data)

        # Update the BPM label
        bpm = float(self.left_channel_sc.bpm)
        self.left_bpm_label.setText(f"LBPM: {bpm:.2f}")

    def update_right_channel_ui(self):
        self.right_pdataitem.setData(self.right_channel_sc.data)

        # Update the BPM label
        bpm = float(self.right_channel_sc.bpm)
        self.right_bpm_label.setText(f"RBPM: {bpm:.2f}")
