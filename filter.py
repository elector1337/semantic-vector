import cv2
import numpy as np
from camera import Camera
from constants import (
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    CENTERLINE_ROW_STEP,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    LOWER_RED1,
    LOWER_RED2,
    MIN_CONTOUR_AREA,
    MORPH_KERNEL,
    RED_CONTOUR_AREA_THRESHOLD,
    RED_MASK_LOWER1,
    RED_MASK_LOWER2,
    RED_MASK_UPPER1,
    RED_MASK_UPPER2,
    DIST,
    MTX,
    ROI_HEIGHT_RATIO,
    S_HSV_ALPHA,
    S_HSV_BETA,
    UPPER_RED1,
    UPPER_RED2,
    V_HSV_ALPHA,
    V_HSV_BETA,
)
import logging
logger = logging.getLogger(__name__)


# класс предобработки
class FrameProcessor:
    """Захватывает кадр и выполняет всю обработку изображения для робота."""

    def __init__(
        self,
        camera: Camera,
        camera_matrix=MTX,
        distortion=DIST,
    ):
        if camera.get_camera() is None:
            raise ValueError("Передан неоткрытый объект Camera.")
        self.camera = camera

        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.new_camera_matrix = None

    def _preprocess(self, frame):
        """Повышает читаемость кадра: шумоподавление, контраст и насыщенность."""
        filtered = cv2.bilateralFilter(frame, d=BILATERAL_D, sigmaColor=BILATERAL_SIGMA_COLOR, sigmaSpace=BILATERAL_SIGMA_SPACE)

        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_clahe = self.clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        balanced_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        s_enhanced = cv2.convertScaleAbs(s, alpha=S_HSV_ALPHA, beta=S_HSV_BETA)
        v_enhanced = cv2.convertScaleAbs(v, alpha=V_HSV_ALPHA, beta=V_HSV_BETA)

        final_hsv = cv2.merge((h, s_enhanced, v_enhanced))

        final_frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return final_frame

    def _undistort(self, frame):
        """Исправляет геометрические искажения камеры при наличии калибровки."""
        if self.camera_matrix is None or self.distortion is None:
            return frame

        height, width = frame.shape[:2]
        if self.new_camera_matrix is None:
            self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                self.camera_matrix, self.distortion, (width, height), 1, (width, height)
            )
        return cv2.undistort(frame, self.camera_matrix, self.distortion, None, self.new_camera_matrix)

    @staticmethod
    def _red_mask(frame):
        """Строит очищенную бинарную маску красной линии."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, RED_MASK_LOWER1, RED_MASK_UPPER1)
        mask2 = cv2.inRange(hsv, RED_MASK_LOWER2, RED_MASK_UPPER2)
        mask = cv2.bitwise_or(mask1, mask2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)
        return mask

    @staticmethod
    def _highlight_red(frame, mask):
        """Оставляет красные объекты цветными и обводит крупные контуры."""

        red_part = cv2.bitwise_and(frame, frame, mask=mask)
        pure_red = np.full_like(frame, (0, 0, 255), dtype=np.uint8)
        enhanced_red = cv2.addWeighted(red_part, 0.3, pure_red, 0.7, 0)
        enhanced_red = cv2.bitwise_and(enhanced_red, enhanced_red, mask=mask)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        background = cv2.bitwise_and(gray_bgr, gray_bgr, mask=cv2.bitwise_not(mask))
        result = cv2.add(enhanced_red, background)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > RED_CONTOUR_AREA_THRESHOLD:
                x, y, width, height = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(result, "Red Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return result

    @staticmethod
    def _extract_centerline(mask, robot_point):
        """Возвращает осевые точки линии от робота в сторону верхней границы кадра."""
        robot_x, robot_y = robot_point
        previous_x = robot_x
        points = []

        for y in range(robot_y - 1, -1, -CENTERLINE_ROW_STEP):
            xs = np.flatnonzero(mask[y])
            if xs.size == 0:
                continue

            # В строке может быть несколько красных объектов. Выбираем сегмент,
            # непрерывнее всего продолжающий линию с предыдущей строки.
            split_indices = np.where(np.diff(xs) > 1)[0] + 1
            segments = np.split(xs, split_indices)
            centers = [float((segment[0] + segment[-1]) / 2) for segment in segments]
            x = min(centers, key=lambda center: abs(center - previous_x))
            points.append((x, float(y)))
            previous_x = x

        return np.asarray(points, dtype=np.float32)

    def get_local_path(self):
        """Возвращает кадр и осевую линию для локального планировщика."""
        ret, frame = self.camera.read()
        if not ret:
            return None, np.empty((0, 2), dtype=np.float32)

        frame = self._undistort(self._preprocess(frame))
        mask = self._red_mask(frame)
        result = self._highlight_red(frame, mask)
        robot_point = (result.shape[1] // 2, result.shape[0] - 1)
        centerline = self._extract_centerline(mask, robot_point)

        for point in centerline.astype(int):
            cv2.circle(result, tuple(point), 2, (255, 255, 0), -1)
        cv2.circle(result, robot_point, 5, (255, 0, 0), -1)
        return result, centerline

    def get_frame_data(self):
        """Совместимый интерфейс: возвращает ближайшую осевую точку линии."""
        result, centerline = self.get_local_path()
        if result is None or len(centerline) == 0:
            return result, False, 0, 0

        cx = int(centerline[0, 0])
        return result, True, cx, result.shape[1] // 2 - cx

    def release(self):
        """Камера принадлежит приложению-владельцу и освобождается там."""
        return None


if __name__ == "__main__":
    camera = Camera()
    if not camera.capture(1):
        raise RuntimeError("Камера не найдена!")
    cap = FrameProcessor(camera)
    while True:
        frame, _, _, _ = cap.get_frame_data()
        if frame is None:
            continue
        cv2.imshow("Result (Red on Grayscale)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    camera.release()
    cv2.destroyAllWindows()
