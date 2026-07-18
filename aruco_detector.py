import cv2
import numpy as np
import time
from history import HistoryManager, ObjectClass, ObjectDetection
from generator import DatasetGenerator, PointsStorage
from constants import ARUCO_SIZE, CAMERA_MATRIX, DIST_COEFFS, ROBOT_ID

IP = ROBOT_ID
class ArucoDetector:
    def __init__(self, ip):
        # запускаем детектор ArUCo
        self.ip = ip
        self.frame = 0
        self.cam = None
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.history = HistoryManager(history_length=30)
        self.storage = PointsStorage()
        self.generator = DatasetGenerator(input_window=20, output_window=10)
        self.object_points = np.array([[ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
                                [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
                                [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
                                [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0]], dtype=np.float32)
    
    def _start_camera(self):
        self.cam = cv2.VideoCapture(f"http://{self.ip}:5010/video_feed")  # подключаемся к камере
        if not self.cam.isOpened():
            raise RuntimeError("Не удалось открыть камеру")
        
    def get_detections(self):
        if self.cam is None:
            self._start_camera()

        ret, frame = self.cam.read()
        if not ret:
            raise RuntimeError("Не удалось получить кадр")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is not None:
            detections = []
            for i in range(len(ids)):
                success, rvec, tvec = cv2.solvePnP(self.object_points, corners[i][0], CAMERA_MATRIX, DIST_COEFFS)

                if success:
                    detection = ObjectDetection(
                    frame=self.frame,
                    marker_id=int(ids[i]),
                    object_class=ObjectClass.STATIC,
                    position=np.array([
                        tvec[0][0],   # X
                        tvec[2][0]    # Z
                        ]),
                    timestamp=time.time()
                    )

                    detections.append(detection)
                    # центр маркера
                    center = np.mean(corners[i][0], axis=0).astype(int)
                    cv2.circle(frame, tuple(center), 5, (0, 0, 255), -1)

                    # подпись
                    text = (
                        f"ID:{detection.marker_id} "
                        f"X:{detection.position[0]:.1f} "
                        f"Z:{detection.position[1]:.1f}"
                    )
                    cv2.putText(
                        frame,
                        text,
                        (center[0] + 10, center[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                    # оси координат (очень полезно для проверки solvePnP)
                    cv2.drawFrameAxes(
                        frame,
                        CAMERA_MATRIX,
                        DIST_COEFFS,
                        rvec,
                        tvec,
                        ARUCO_SIZE/2

                    )
            self.history.update(detections)
            for detection in detections:
                self.storage.append(detection)
        else:
            cv2.putText(frame, "No markers detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.frame += 1
        cv2.imshow("Robot Camera View", frame)

    def save_dataset(self):
        self.storage.save()
        X, Y = self.generator.generate(self.storage.df)
        self.generator.save("dataset.npz", X, Y)
        self.cam.release()

if __name__ == "__main__":
    detector = ArucoDetector(IP)
    while True:
        detector.get_detections()
        if cv2.waitKey(1) & 0xFF == ord('q'):
            detector.save_dataset()
            break

cv2.destroyAllWindows()
