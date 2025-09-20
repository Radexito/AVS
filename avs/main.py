"""
Author: Radosław Wysocki
Contact: wysockiradek@gmail.com
"""
import logging

import pyaudio
import sys

from PySide6 import QtWidgets
from PySide6.QtGui import Qt

from avs.widgets.main_window import MainWindow


def select_microphone_input():
    # TODO: Remove this function when implemented config.
    p = pyaudio.PyAudio()
    microphones = []

    # Get a list of available microphones
    for i in range(p.get_device_count()):
        device_info = p.get_device_info_by_index(i)
        if device_info['maxInputChannels'] > 0:
            microphones.append(device_info)

    print("Available Microphones:")
    for i, mic in enumerate(microphones):
        print(f"{i + 1}. {mic['name']}")

    while True:
        choice = input("Select a microphone by entering its corresponding number: ")
        try:
            choice = int(choice)
            if 1 <= choice <= len(microphones):
                selected_microphone = microphones[choice - 1]
                return selected_microphone
            else:
                print("Invalid choice. Please enter a valid microphone number.")
        except ValueError:
            print("Invalid input. Please enter a valid microphone number.")


def main():
    logging.basicConfig(level=10, filename="log.txt")
    left_channel_index = select_microphone_input()
    right_channel_index = select_microphone_input()
    app = QtWidgets.QApplication(sys.argv)
    main_window = MainWindow(left_channel_index, right_channel_index)
    main_window.show()
    main_window.setWindowState(Qt.WindowActive)
    main_window.raise_()
    main_window.activateWindow()

    app.exec_()


if __name__ == '__main__':
    main()
