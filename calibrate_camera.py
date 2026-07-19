from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from constants import ROBOT_VIDEO_STREAM_URL
from pc_vision import load_intrinsics, undistort, load_homography


class CalibrationWindow:
    def __init__(self, source: int | str = 0) -> None:
        if source in (None, 0, "0", "robot", "url"):
            source_value = ROBOT_VIDEO_STREAM_URL
        elif isinstance(source, str) and (source.startswith("http://") or source.startswith("https://")):
            source_value = source
        else:
            source_value = int(source)

        self.cap = cv2.VideoCapture(source_value)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open camera {source_value}")
        self.frame = None
        self.successes: List[Tuple[List[np.ndarray], List[np.ndarray]]] = []
        self.points: List[Tuple[Tuple[int, int], ...]] = []

    def read_frame(self) -> Optional[np.ndarray]:
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.frame = frame
        return frame

    def release(self) -> None:
        self.cap.release()


def save_intrinsics(output_path: str, matrix: np.ndarray, distortion: np.ndarray, rms_error: float) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, matrix=matrix, distortion=distortion, rms_error=np.array(rms_error, dtype=np.float32))


def save_homography(output_path: str, homography: np.ndarray, size: Tuple[int, int]) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, homography=homography, size=np.array(size, dtype=np.int32))


def calibrate_intrinsics(args: argparse.Namespace) -> None:
    window = CalibrationWindow(args.source)
    pattern_size = (args.columns, args.rows)
    board_size = args.square_size
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print("Place the chessboard in view, then press 'S' to save a frame, 'C' to calibrate, 'Q' to quit.")

    while True:
        frame = window.read_frame()
        if frame is None:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        display = frame.copy()
        if found:
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display, pattern_size, corners, found)

        cv2.putText(display, f"Samples: {len(window.successes)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("Chessboard Calibration", display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s') and found:
            window.successes.append(corners)
            print(f"Saved sample {len(window.successes)}")
        if key == ord('c') and len(window.successes) >= 10:
            objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2) * board_size
            obj_points = []
            img_points = []
            for corners in window.successes:
                obj_points.append(objp)
                img_points.append(corners)

            rms_error, matrix, distortion, _, _ = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
            save_intrinsics(args.output, matrix, distortion, float(rms_error))
            print(f"Calibration complete. RMS={rms_error:.5f}")
            break

    window.release()
    cv2.destroyAllWindows()


def calibrate_homography(args: argparse.Namespace) -> None:
    intrinsics = load_intrinsics(args.intrinsics)
    window = CalibrationWindow(args.source)
    points: List[Tuple[Tuple[int, int], ...]] = []

    print("Click the rectangle corners in order: near-left, near-right, far-right, far-left")

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

    cv2.namedWindow("Homography Calibration")
    cv2.setMouseCallback("Homography Calibration", on_click)

    while True:
        frame = window.read_frame()
        if frame is None:
            break
        undistorted = undistort(frame, intrinsics[0], intrinsics[1])
        display = undistorted.copy()

        if points:
            for idx, point in enumerate(points):
                cv2.circle(display, point, 5, (0, 255, 0), -1)
                cv2.putText(display, str(idx + 1), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(display, "Press Enter to save homography", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Homography Calibration", display)
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        if key == 13 and len(points) == 4:
            src = np.array(points, dtype=np.float32)
            width_px = int(args.width_m * args.pixels_per_meter)
            height_px = int(args.height_m * args.pixels_per_meter)
            dst = np.array([
                [0, height_px],
                [width_px, height_px],
                [width_px, 0],
                [0, 0],
            ], dtype=np.float32)
            homography = cv2.getPerspectiveTransform(src, dst)
            save_homography(args.output, homography, (width_px, height_px))
            print("Homography saved")
            break

    window.release()
    cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Camera calibration utilities")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    intrinsics_parser = subparsers.add_parser("intrinsics", help="Calibrate camera intrinsics")
    intrinsics_parser.add_argument("--source", default=0)
    intrinsics_parser.add_argument("--columns", type=int, default=9)
    intrinsics_parser.add_argument("--rows", type=int, default=6)
    intrinsics_parser.add_argument("--square-size", type=float, default=0.025)
    intrinsics_parser.add_argument("--output", required=True)

    homography_parser = subparsers.add_parser("homography", help="Calibrate homography")
    homography_parser.add_argument("--source", default=0)
    homography_parser.add_argument("--intrinsics", required=True)
    homography_parser.add_argument("--width-m", type=float, default=0.6)
    homography_parser.add_argument("--height-m", type=float, default=1.0)
    homography_parser.add_argument("--pixels-per-meter", type=float, default=1000.0)
    homography_parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "intrinsics":
        calibrate_intrinsics(args)
    elif args.mode == "homography":
        calibrate_homography(args)
    else:
        parser.error("Unsupported mode")


if __name__ == "__main__":
    main()
