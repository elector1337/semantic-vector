from __future__ import annotations

import argparse
import cv2
import numpy as np
from scipy.interpolate import splev

from constants import ROBOT_VIDEO_STREAM_URL

from pc_vision import (
    DEFAULT_FILTER_CONFIG,
    detect_aruco,
    draw_path,
    extract_centerline,
    fit_spline,
    load_filter_config,
    load_homography,
    load_intrinsics,
    red_mask,
    save_filter_config,
    transform_points,
    undistort,
    warp_bird_view,
)


FILTER_CONFIG_PATH = "calibration/filter_config.json"
FILTER_WINDOW = "Filter Controls"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Vision viewer for the semantic vector pipeline")
    parser.add_argument("--source", default=ROBOT_VIDEO_STREAM_URL)
    parser.add_argument("--intrinsics", default=None)
    parser.add_argument("--homography", default=None)
    parser.add_argument("--filter-config", default=FILTER_CONFIG_PATH)
    return parser


def normalize_source(source: str):
    if source in ("robot", "url"):
        return ROBOT_VIDEO_STREAM_URL
    if source.isdigit():
        return int(source)
    return source


def draw_mode_label(frame: np.ndarray, text: str) -> np.ndarray:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(output, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return output


def red_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = frame.copy()
    output[mask > 0] = (0, 0, 255)
    return cv2.addWeighted(frame, 0.65, output, 0.35, 0)


def setup_filter_controls(config: dict) -> None:
    cv2.namedWindow(FILTER_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(FILTER_WINDOW, 420, 520)
    ranges = {
        "h1_min": 179,
        "h1_max": 179,
        "h2_min": 179,
        "h2_max": 179,
        "s_min": 255,
        "v_min": 255,
        "r_min": 255,
        "rg_min": 120,
        "rb_min": 120,
        "excess_min": 120,
        "soft_rg_min": 120,
        "open_size": 31,
        "close_size": 31,
        "min_area": 5000,
    }
    for key, maximum in ranges.items():
        cv2.createTrackbar(key, FILTER_WINDOW, int(config.get(key, DEFAULT_FILTER_CONFIG[key])), maximum, lambda value: None)


def read_filter_controls() -> dict:
    config = {}
    for key in DEFAULT_FILTER_CONFIG:
        config[key] = cv2.getTrackbarPos(key, FILTER_WINDOW)
    config["open_size"] = max(1, config["open_size"] | 1)
    config["close_size"] = max(1, config["close_size"] | 1)
    return config


def apply_filter_controls(config: dict) -> None:
    for key, value in config.items():
        if key in DEFAULT_FILTER_CONFIG:
            cv2.setTrackbarPos(key, FILTER_WINDOW, int(value))


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    source = normalize_source(str(args.source))
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError("Unable to open camera")

    intrinsics = None
    homography = None
    homography_size = None
    if args.intrinsics:
        intrinsics = load_intrinsics(args.intrinsics)
    if args.homography:
        homography, homography_size = load_homography(args.homography)

    filter_config = load_filter_config(args.filter_config)
    setup_filter_controls(filter_config)
    current_mode = '1'

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        undistorted = undistort(frame, intrinsics[0] if intrinsics else None, intrinsics[1] if intrinsics else None)
        if homography is not None:
            bird_view = warp_bird_view(undistorted, homography, homography_size or (600, 1000))
        else:
            bird_view = None

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord('s'):
            save_filter_config(args.filter_config, read_filter_controls())
            print(f"Saved filter config to {args.filter_config}")
        if key == ord('r'):
            apply_filter_controls(DEFAULT_FILTER_CONFIG)
            print("Reset filter controls to defaults")
        if key in {ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6')}:
            current_mode = chr(key)

        filter_config = read_filter_controls()
        mode_label = "1 raw undistorted"

        if current_mode == '1':
            view = undistorted
        elif current_mode == '2':
            mask = red_mask(undistorted, keep_largest=False, config=filter_config)
            view = red_overlay(undistorted, mask)
            mode_label = "2 red filter overlay"
        elif current_mode == '3':
            if bird_view is not None:
                mask = red_mask(bird_view, keep_largest=False, config=filter_config)
                view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            else:
                mask = red_mask(undistorted, keep_largest=False, config=filter_config)
                view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            mode_label = "3 red mask"
        elif current_mode == '4':
            if bird_view is not None:
                mask = red_mask(bird_view, config=filter_config)
                points = extract_centerline(mask)
                spline = fit_spline(points) if len(points) >= 3 else None
                if spline is not None:
                    tck, u = spline
                    sample_u = np.linspace(u[0], u[-1], 200)
                    sample_points = np.asarray(np.array([np.array(splev(sample_u, tck)[0]), np.array(splev(sample_u, tck)[1])]).T, dtype=np.float32)
                    projected_points = transform_points(sample_points, homography, inverse=True)
                    view = draw_path(undistorted, transform_points(points, homography, inverse=True), None)
                    for index in range(1, len(projected_points)):
                        a = tuple(map(int, projected_points[index - 1]))
                        b = tuple(map(int, projected_points[index]))
                        cv2.line(view, a, b, (255, 0, 0), 2)
                else:
                    view = undistorted
            else:
                view = undistorted
            mode_label = "4 real view + bird spline"
        elif current_mode == '5':
            if bird_view is not None:
                mask = red_mask(bird_view, config=filter_config)
                points = extract_centerline(mask)
                spline = fit_spline(points) if len(points) >= 3 else None
                view = draw_path(bird_view, points, spline)
            else:
                mask = red_mask(undistorted, config=filter_config)
                points = extract_centerline(mask)
                spline = fit_spline(points) if len(points) >= 3 else None
                view = draw_path(undistorted, points, spline)
            mode_label = "5 bird view + spline"
        elif current_mode == '6':
            view = detect_aruco(undistorted, intrinsics)
            mode_label = "6 aruco markers"

        if view is None:
            view = undistorted

        view = draw_mode_label(view, mode_label)
        cv2.imshow("Semantic Vector Viewer", view)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
