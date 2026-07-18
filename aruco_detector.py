import logging
import cv2
import numpy as np
import time
from history import ObjectClass, ObjectDetection

logger = logging.getLogger(__name__)
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
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.parameters.adaptiveThreshWinSizeMin = 5
        self.parameters.adaptiveThreshWinSizeMax = 31
        self.parameters.adaptiveThreshWinSizeStep = 4
        self.parameters.minMarkerPerimeterRate = 0.02
        self.parameters.maxMarkerPerimeterRate = 4.0

        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)
        self.object_points = np.array([[ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0],
                                [ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
                                [-ARUCO_SIZE / 2, -ARUCO_SIZE / 2, 0],
                                [-ARUCO_SIZE / 2, ARUCO_SIZE / 2, 0]], dtype=np.float32)
    
    def _start_camera(self):
        self.cam = cv2.VideoCapture(f"http://{self.ip}/video_feed")  # подключаемся к камере
        if not self.cam.isOpened():
            logger.critical("camera nor found")
            raise RuntimeError("Не удалось открыть камеру")
        
    def get_detections(self):
        if self.cam is None:
            self._start_camera()

        ret, frame = self.cam.read()
        if not ret:
                self.cam.release()
                self.cam = None
                return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
        detections = []
        
        if ids is not None:
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
                        f"y:{detection.position[1]:.1f}"
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
        else:
            cv2.putText(frame, "No markers detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        self.frame += 1
        cv2.imshow("Robot Camera View", frame)
        return detections

    def stop(self):
        self.cam.release()
        logger.info("camera stopped")
        cv2.destroyAllWindows()
