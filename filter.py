from __future__ import annotations

import cv2
import numpy as np

from constants import (
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    MIN_CONTOUR_AREA,
    ROBOT_VIDEO_STREAM_URL,
    ROI_HEIGHT_RATIO,
)
from pc_vision import load_filter_config, red_mask


class CameraPreprocessor:
    def __init__(self, camera_index=0, width=DEFAULT_CAMERA_WIDTH, height=DEFAULT_CAMERA_HEIGHT, filter_config_path="calibration/filter_config.json"):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.filter_config_path = filter_config_path

        if not self.cap.isOpened():
            raise RuntimeError(f"Camera is not available: {camera_index}")

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        return frame

    def get_processed_hsv(self):
        frame = self.read()
        if frame is None:
            return None, None
        return frame, cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def get_mask(self):
        frame = self.read()
        if frame is None:
            return None, None
        return frame, red_mask(frame, config=load_filter_config(self.filter_config_path))

    def release(self):
        self.cap.release()


def get_line_position(frame, center_x):
    height, _ = frame.shape[:2]
    mask = red_mask(frame)

    roi_y_start = int(height * (1 - ROI_HEIGHT_RATIO))
    roi = mask[roi_y_start:height, :]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False, 0, 0, mask, roi_y_start

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        return False, 0, 0, mask, roi_y_start

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        return False, 0, 0, mask, roi_y_start

    cx = int(moments["m10"] / moments["m00"])
    error = center_x - cx
    return True, cx, error, mask, roi_y_start


def main():
    cap = CameraPreprocessor(ROBOT_VIDEO_STREAM_URL)
    try:
        while True:
            frame, mask = cap.get_mask()
            if frame is None:
                break

            overlay = frame.copy()
            overlay[mask > 0] = (0, 0, 255)
            result = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)

            cv2.imshow("Original", frame)
            cv2.imshow("Red Mask", mask)
            cv2.imshow("Red Overlay", result)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
