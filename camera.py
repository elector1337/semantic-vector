import cv2
from threading import Lock

class Camera:
    def __init__(self):
        self._camera = None
        self._lock = Lock()

    def capture(self, index, width=640, height=480):
        """Открывает камеру и задаёт параметры видеопотока."""
        self.release()
        self._camera = cv2.VideoCapture(index)
        self._camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return self._camera.isOpened()

    def get_camera(self):
        return self._camera

    def read(self):
        """Безопасно читает один кадр из единственного видеопотока."""
        with self._lock:
            if self._camera is None:
                return False, None
            return self._camera.read()
    
    def release(self):
        with self._lock:
            if self._camera is not None:
                self._camera.release()
                self._camera = None
