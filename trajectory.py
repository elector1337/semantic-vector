import cv2
import numpy as np
from filter import CameraPreprocessor

# Camera/frame sizing and scanning parameters
x_camera_size = 640
y_camera_size = 480
num_strips = 20
strip_height = y_camera_size // num_strips

# Default HSV ranges for red (two ranges to cover hue wraparound)
lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([180, 255, 255])

# Углы для перспективного преобразования (можно переопределить)
src_crn = np.float32([[200., 300.], [440., 300.], [640., 480.], [0., 480.]])
out_ctn = np.float32([[0., 0.], [640., 0.], [640., 480.], [0., 480.]])
matrix = cv2.getPerspectiveTransform(src_crn, out_ctn)


def detect_red_points(frame, perspective_matrix=None, num_strips=num_strips, y_camera_size=y_camera_size,
                      lower1=lower_red1, upper1=upper_red1, lower2=lower_red2, upper2=upper_red2,
                      draw=True):
    """
    Detect approximate red points by scanning horizontal strips.

    Args:
        frame: BGR image (numpy array).
        perspective_matrix: optional 3x3 matrix to warp the frame before detection.
        num_strips: number of horizontal strips to scan.
        y_camera_size: vertical size used to compute strip height.
        lower1/upper1, lower2/upper2: HSV bounds for red mask.
        draw: if True, returns a visualization image with detected points drawn.

    Returns:
        detected_points: list of [x, y] points (one per strip where found).
        vis_frame: visualization BGR image if draw True else None.
        red_mask: binary mask used for detection.
    """
    if perspective_matrix is not None:
        output = cv2.warpPerspective(frame, perspective_matrix, (x_camera_size, y_camera_size))
        gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
        if cv2.countNonZero(gray) < output.shape[0] * output.shape[1] * 0.02:
            # fallback: если вырезанная область пуста, используем исходное изображение
            output = frame.copy()
    else:
        output = frame.copy()

    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    strip_h = y_camera_size // num_strips
    detected_points = []
    vis_frame = output.copy() if draw else None

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)

    for i in range(num_strips):
        y_start = i * strip_h
        y_end = (i + 1) * strip_h

        strip_mask = red_mask[y_start:y_end, :]
        moments = cv2.moments(strip_mask)

        if moments.get("m00", 0) > 0:
            cx = int(moments["m10"] / moments["m00"]) if moments["m00"] != 0 else 0
            cy = int(moments["m01"] / moments["m00"]) + y_start if moments["m00"] != 0 else y_start
            detected_points.append([cx, cy])

            if draw and vis_frame is not None:
                cv2.circle(vis_frame, (cx, cy), 5, (0, 255, 0), -1)
                cv2.putText(vis_frame, f"{i}", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (0, 255, 0), 1)

    return detected_points, vis_frame, red_mask


if __name__ == '__main__':
    # Live mode using CameraPreprocessor (preserve original behaviour)
    cam = CameraPreprocessor("http://192.168.2.2:5000/video_feed")
    
    while True:
        original_frame, processed_frame = cam.get_processed_hsv()

        if original_frame is None:
            break

        # Use processed frame for detection (already enhanced by CameraPreprocessor)
        detected_points, vis_frame, red_mask = detect_red_points(processed_frame, perspective_matrix=matrix)

        #cv2.imshow("Original", original_frame)
        cv2.imshow("Red Mask", red_mask)
        if vis_frame is not None:
            cv2.imshow("Output with Points", vis_frame)

        print(f"Найдено точек: {len(detected_points)}")
        print("Координаты:", detected_points)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()