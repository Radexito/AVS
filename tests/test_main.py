from unittest import TestCase
from unittest.mock import patch, MagicMock

from avs import main


class TestMain(TestCase):
    @patch.object(main, "select_microphone_input")
    @patch.object(main, "MainWindow")
    @patch.object(main.QtWidgets.QApplication, "exec_")
    def test_main_window(self, exec_: MagicMock, main_window: MagicMock, select_microphone_input: MagicMock):
        # Mock
        select_microphone_input.return_value = 0

        # Act
        main.main()

        # Assert
        with self.subTest("MainWindow instantiated"):
            main_window.assert_called()

        with self.subTest("MainWindow show called"):
            main_window().show.assert_called()

        with self.subTest("app exec_ called"):
            exec_.assert_called()
