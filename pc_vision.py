from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    from scipy.interpolate import splprep, splev
except ImportError:  # pragma: no cover - optional dependency
    splprep = None
    splev = None


DEFAULT_FILTER_CONFIG = {
    "h1_min": 0,
    "h1_max": 12,
    "h2_min": 165,
    "h2_max": 179,
    "s_min": 45,
    "v_min": 55,
    "r_min": 65,
    "rg_min": 22,
    "rb_min": 14,
    "excess_min": 18,
    "soft_rg_min": 28,
    "open_size": 3,
    "close_size": 9,
    "min_area": 80,
}


def normalized_filter_config(config: Optional[dict] = None) -> dict:
    output = DEFAULT_FILTER_CONFIG.copy()
    if config:
        output.update({key: int(value) for key, value in config.items() if key in output})
    output["open_size"] = max(1, output["open_size"] | 1)
    output["close_size"] = max(1, output["close_size"] | 1)
    return output


def load_filter_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return DEFAULT_FILTER_CONFIG.copy()
    with config_path.open("r", encoding="utf-8") as file:
        return normalized_filter_config(json.load(file))


def save_filter_config(path: str, config: dict) -> None:
    config_path = Path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as file:
        json.dump(normalized_filter_config(config), file, indent=2)


def load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load camera matrix and distortion coefficients from an .npz file."""
    data = np.load(path)
    matrix = None
    distortion = None

    for key in ("matrix", "camera_matrix", "mtx"):
        if key in data:
            matrix = data[key]
            break

    for key in ("distortion", "dist_coeffs", "dist"):
        if key in data:
            distortion = data[key]
            break

    if matrix is None or distortion is None:
        raise KeyError(f"Intrinsics file {path} must contain matrix and distortion arrays")

    return matrix.astype(np.float32), distortion.astype(np.float32)


def undistort(frame: np.ndarray, matrix: Optional[np.ndarray], distortion: Optional[np.ndarray]) -> np.ndarray:
    """Apply lens distortion correction to a frame."""
    if frame is None:
        return None
    if matrix is None or distortion is None:
        return frame.copy()

    h, w = frame.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 1, (w, h))
    return cv2.undistort(frame, matrix, distortion, None, new_camera_matrix)


def load_homography(path: str) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Load homography matrix and output size from an .npz file."""
    data = np.load(path)

    homography = None
    size = None

    for key in ("homography", "H"):
        if key in data:
            homography = data[key]
            break

    for key in ("size", "output_size"):
        if key in data:
            size = data[key]
            break

    if homography is None:
        raise KeyError(f"Homography file {path} must contain a homography matrix")

    if size is None:
        size = (600, 1000)
    else:
        size = tuple(int(v) for v in size)

    return homography.astype(np.float32), size


def warp_bird_view(frame: np.ndarray, homography: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """Warp an image into bird's-eye view using the supplied homography."""
    if frame is None:
        return None
    if homography is None:
        return frame.copy()
    bird_view = cv2.warpPerspective(frame, homography, size)
    if bird_view.shape[0] < frame.shape[0] // 2:
        scale = max(1.0, frame.shape[0] / float(bird_view.shape[0]))
        bird_view = cv2.resize(bird_view, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    return bird_view


def transform_points(points: np.ndarray, homography: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Apply a perspective transform to a set of 2D points."""
    if points is None or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)

    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    if homography is None:
        return pts.reshape(-1, 2)

    if inverse:
        homography = np.linalg.inv(homography)

    transformed = cv2.perspectiveTransform(pts, homography)
    return transformed.reshape(-1, 2)


def red_mask(frame: np.ndarray, keep_largest: bool = True, config: Optional[dict] = None) -> np.ndarray:
    """Create a stable binary mask for the red line."""
    if frame is None:
        return None

    if frame.ndim == 2:
        return cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY)[1]

    config = normalized_filter_config(config)
    blurred = cv2.GaussianBlur(frame, (3, 3), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(blurred)

    lower_red1 = np.array([config["h1_min"], config["s_min"], config["v_min"]], dtype=np.uint8)
    upper_red1 = np.array([config["h1_max"], 255, 255], dtype=np.uint8)
    lower_red2 = np.array([config["h2_min"], config["s_min"], config["v_min"]], dtype=np.uint8)
    upper_red2 = np.array([config["h2_max"], 255, 255], dtype=np.uint8)

    hue_mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))

    r16 = r.astype(np.int16)
    g16 = g.astype(np.int16)
    b16 = b.astype(np.int16)
    red_excess = r16 - np.maximum(g16, b16)
    dominance = ((r16 - g16) > config["rg_min"]) & ((r16 - b16) > config["rb_min"]) & (r > config["r_min"])
    soft_red = (red_excess > config["excess_min"]) & ((r16 - g16) > config["soft_rg_min"]) & (r > config["r_min"])
    mask = cv2.bitwise_or(
        cv2.bitwise_and(hue_mask, dominance.astype(np.uint8) * 255),
        soft_red.astype(np.uint8) * 255,
    )

    open_kernel = np.ones((config["open_size"], config["open_size"]), np.uint8)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config["close_size"], config["close_size"]))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    if not keep_largest:
        return mask

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= config["min_area"]]
    if not contours:
        return np.zeros(mask.shape, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    filtered = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(filtered, [largest], -1, 255, thickness=cv2.FILLED)
    mask = filtered

    return mask


def extract_centerline(mask: np.ndarray) -> np.ndarray:
    """Extract a single centerline from a binary mask by following the strongest segment per row."""
    if mask is None:
        return np.empty((0, 2), dtype=np.float32)

    height, width = mask.shape[:2]
    points = []
    previous_center = None

    for y in range(height - 1, -1, -1):
        row = mask[y]
        segments = []
        start = None
        for x in range(width):
            if row[x] > 0:
                if start is None:
                    start = x
            elif start is not None:
                segments.append((start, x - 1))
                start = None
        if start is not None:
            segments.append((start, width - 1))

        if not segments:
            continue

        if previous_center is None:
            segment = max(segments, key=lambda item: item[1] - item[0])
            center_x = (segment[0] + segment[1]) / 2.0
            previous_center = center_x
            points.append([center_x, float(y)])
            continue

        best_segment = None
        best_distance = float("inf")
        for segment in segments:
            center_x = (segment[0] + segment[1]) / 2.0
            distance = abs(center_x - previous_center)
            if distance < best_distance:
                best_distance = distance
                best_segment = segment

        if best_segment is not None and best_distance < max(60.0, width * 0.25):
            center_x = (best_segment[0] + best_segment[1]) / 2.0
            previous_center = center_x
            points.append([center_x, float(y)])

    if not points:
        return np.empty((0, 2), dtype=np.float32)

    points = np.asarray(points, dtype=np.float32)
    if len(points) > 1:
        points = points[::-1]
    return points


def fit_spline(points: np.ndarray) -> Optional[Tuple[object, np.ndarray]]:
    """Fit a smoothed local B-spline to a set of image points."""
    if splprep is None or splev is None:
        return None

    if points is None or len(points) < 3:
        return None

    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2:
        return None

    try:
        tck, u = splprep([points[:, 0], points[:, 1]], s=max(1.0, 0.001 * len(points)), k=min(3, len(points) - 1))
        return tck, u
    except Exception:
        return None


def draw_path(frame: np.ndarray, points: np.ndarray, spline: Optional[Tuple[object, np.ndarray]] = None) -> np.ndarray:
    """Draw a sequence of path points and an optional spline onto a frame."""
    if frame is None:
        return None

    output = frame.copy()
    if points is not None and len(points) > 1:
        points = np.asarray(points, dtype=np.float32)
        for index, point in enumerate(points):
            x, y = int(point[0]), int(point[1])
            cv2.circle(output, (x, y), 2, (0, 255, 255), -1)
            if index > 0:
                prev_x, prev_y = int(points[index - 1][0]), int(points[index - 1][1])
                cv2.line(output, (prev_x, prev_y), (x, y), (0, 255, 255), 1)

    if spline is not None and splev is not None:
        tck, u = spline
        if u is not None and len(u) > 2:
            sample_u = np.linspace(u[0], u[-1], 200)
            sample_points = np.asarray(splev(sample_u, tck), dtype=np.float32).T
            for index in range(1, len(sample_points)):
                a = tuple(map(int, sample_points[index - 1]))
                b = tuple(map(int, sample_points[index]))
                cv2.line(output, a, b, (255, 0, 0), 2)

    return output


def detect_aruco(frame: np.ndarray, intrinsics: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
    """Detect ArUco markers, draw their boxes, ids, and axes."""
    if frame is None:
        return None

    matrix = None
    distortion = None
    if intrinsics is not None:
        matrix, distortion = intrinsics

    output = frame.copy()
    gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

    try:
        aruco_module = cv2.aruco
    except AttributeError:  # pragma: no cover - fallback for some builds
        aruco_module = None

    if aruco_module is None:
        cv2.putText(output, "cv2.aruco unavailable", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return output

    dictionary = aruco_module.getPredefinedDictionary(aruco_module.DICT_5X5_100)
    parameters = aruco_module.DetectorParameters_create() if hasattr(aruco_module, "DetectorParameters_create") else aruco_module.DetectorParameters()

    if hasattr(aruco_module, "ArucoDetector"):
        detector = aruco_module.ArucoDetector(dictionary, parameters)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = aruco_module.detectMarkers(gray, dictionary, parameters=parameters)

    if ids is not None:
        for idx, marker_corners in enumerate(corners):
            marker_id = int(ids[idx][0])
            marker_corners = marker_corners.reshape(-1, 2).astype(np.int32)
            cv2.polylines(output, [marker_corners], True, (0, 255, 0), 2)
            cv2.putText(output, str(marker_id), (marker_corners[0][0], marker_corners[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if matrix is not None and distortion is not None:
                object_points = np.array([
                    [-0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [-0.5, -0.5, 0.0],
                ], dtype=np.float32)
                success, rvec, tvec = cv2.solvePnP(object_points, marker_corners.reshape(4, 1, 2), matrix, distortion)
                if success:
                    cv2.drawFrameAxes(output, matrix, distortion, rvec, tvec, 0.05)
    else:
        cv2.putText(output, "No markers detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return output
