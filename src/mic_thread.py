import traceback
import sys
from qtpy import QtCore


class MicThread(QtCore.QThread):
    sig = QtCore.Signal(bytes)

    def __init__(self, sc):
        super(MicThread, self).__init__()
        self.sc = sc
        self.sig.connect(self.sc.append)
        self.running = True

    def run(self):
        try:
            while self.running:
                try:
                    data = self.sc.stream.read(self.sc.CHUNK, exception_on_overflow=False)
                    self.sig.emit(data)
                except OSError as e:
                    sys.stdout.write(str(e))
                    sys.stdout.write(str(traceback.format_exc()))
                    break
        except Exception as e:
            sys.stdout.write(str(type(e)))
            sys.stdout.write(str(e))
            sys.stdout.write(str(traceback.format_exc()))

    def stop(self):
        sys.stdout.write('THREAD STOPPED')
        self.running = False