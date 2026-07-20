import logging
import cv2
import numpy as np
import time
from history import ObjectClass, ObjectDetection
from pc_vision import create_aruco_detector, detect_aruco_markers

logger = logging.getLogger(__name__)
from constants import ARUCO_SIZE, CAMERA_MATRIX, DIST_COEFFS, ROBOT_ID

IP = ROBOT_ID
class ArucoDetector:
    def __init__(self, ip):
        # запускаем детектор ArUCo
        self.ip = ip
        self.frame = 0
        self.cam = None
        self.detector = create_aruco_detector()
    
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

        marker_data, debug_frame = detect_aruco_markers(
            frame,
            intrinsics=(CAMERA_MATRIX, DIST_COEFFS),
            marker_size=ARUCO_SIZE,
            detector=self.detector,
        )
        detections = []
        
        for marker in marker_data:
            tvec = marker["tvec"]
            if tvec is None:
                continue

            detection = ObjectDetection(
                    frame=self.frame,
                    marker_id=int(marker["id"]),
                    object_class=ObjectClass.STATIC,
                    position=np.array([
                        tvec[0][0],   # X
                        tvec[2][0]    # Z
                        ]),
                    timestamp=time.time()
            )
            detections.append(detection)

        self.frame += 1
        cv2.imshow("Robot Camera View", debug_frame)
        return detections

    def stop(self):
        self.cam.release()
        logger.info("camera stopped")
        cv2.destroyAllWindows()
