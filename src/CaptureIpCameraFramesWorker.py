from PyQt5.QtCore import QThread, pyqtSignal
import av
import numpy as np
import time

class CaptureIpCameraFramesWorker(QThread):
    ImageUpdated = pyqtSignal(np.ndarray, int)  # Emit numpy.ndarray and label index

    def __init__(self, camera_address, label_index):
        super().__init__()
        self.camera_address = camera_address
        self.label_index = label_index  # Store the label index
        self.running = True
        self.fps_limit = 30  # Set the frame limit per second

    def run(self):
        try:
            container = av.open(self.camera_address)
            stream = container.streams.video[0]
            for packet in container.demux(stream):
                if not self.running:
                    break
                for frame in packet.decode():
                    image = frame.to_ndarray(format='bgr24')
                    self.ImageUpdated.emit(image, self.label_index)
        except Exception as e:
            print(f"Exception in CaptureIpCameraFramesWorker: {e}")

    def stop(self):
        self.running = False
