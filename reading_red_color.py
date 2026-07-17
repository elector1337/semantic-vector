import cv2
import numpy as np

LOWER_RED1 = np.array([0, 120, 100])
UPPER_RED1 = np.array([5, 255, 255])
LOWER_RED2 = np.array([175, 120, 100])
UPPER_RED2 = np.array([180, 255, 255])

DEFAULT_STRIPES = 6
DEFAULT_MIN_CONTOUR_AREA = 80


def build_red_mask(frame, kernel_size=(5, 5)):
    """Построить бинарную маску красного цвета с морфологической очисткой."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones(kernel_size, np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def get_stripe_trajectory(frame, stripes=DEFAULT_STRIPES, min_area=DEFAULT_MIN_CONTOUR_AREA):
    """Разбивает кадр на горизонтальные полосы и возвращает траекторию по красным точкам.

    Каждая точка содержит [x, y, distance, error].
    distance — пиксельное расстояние от нижнего края кадра.
    error — отклонение от центральной вертикали.
    """
    height, width = frame.shape[:2]
    center_x = width // 2
    mask = build_red_mask(frame)

    stripe_height = max(1, height // stripes)
    trajectory = []

    for stripe_index in range(stripes):
        y2 = height - stripe_index * stripe_height
        y1 = max(0, y2 - stripe_height)
        stripe_mask = mask[y1:y2, :]

        contours, _ = cv2.findContours(stripe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) < min_area:
            continue

        M = cv2.moments(best)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"]) + y1
        error = center_x - cx
        distance = float(height - cy)

        trajectory.append([cx, cy, distance, error])

    return trajectory


def get_line_position(frame, center_x, stripes=DEFAULT_STRIPES, min_area=DEFAULT_MIN_CONTOUR_AREA):
    """Найти первую (ближайшую) красную точку по полоскам и вернуть позицию для управления."""
    mask = build_red_mask(frame)
    height = frame.shape[0]
    stripe_height = max(1, height // stripes)

    for stripe_index in range(stripes):
        y2 = height - stripe_index * stripe_height
        y1 = max(0, y2 - stripe_height)
        stripe_mask = mask[y1:y2, :]

        contours, _ = cv2.findContours(stripe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        best = max(contours, key=cv2.contourArea)
        if cv2.contourArea(best) < min_area:
            continue

        M = cv2.moments(best)
        if M["m00"] == 0:
            continue

        cx = int(M["m10"] / M["m00"])
        error = center_x - cx
        return True, cx, error, mask, y1

    return False, 0, 0, mask, height - stripe_height


if __name__ == "__main__":
    from filter import CameraPreprocessor

    stream_url = "http://192.168.2.2:5000/video_feed"
    cap = CameraPreprocessor(stream_url)

    print("Демонстрация траектории по полоскам. Нажми 'q' для выхода.")
    while True:
        original_frame, frame = cap.get_processed_hsv()
        if frame is None:
            break

        trajectory = get_stripe_trajectory(frame, stripes=6, min_area=80)
        debug = frame.copy()
        for x, y, distance, error in trajectory:
            cv2.circle(debug, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(debug, f"d={int(distance)} e={int(error)}", (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imshow("Stripe Trajectory", debug)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()