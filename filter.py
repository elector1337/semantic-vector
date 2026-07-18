import cv2
import numpy as np
from constants import (
    BILATERAL_D,
    BILATERAL_SIGMA_COLOR,
    BILATERAL_SIGMA_SPACE,
    CLAHE_CLIP_LIMIT,
    CLAHE_TILE_GRID_SIZE,
    DEFAULT_CAMERA_HEIGHT,
    DEFAULT_CAMERA_WIDTH,
    ER_INFO,
    LOWER_RED1,
    LOWER_RED2,
    MIN_CONTOUR_AREA,
    MORPH_KERNEL,
    RED_CONTOUR_AREA_THRESHOLD,
    RED_MASK_LOWER1,
    RED_MASK_LOWER2,
    RED_MASK_UPPER1,
    RED_MASK_UPPER2,
    ROBOT_VIDEO_STREAM_URL,
    ROI_HEIGHT_RATIO,
    S_HSV_ALPHA,
    S_HSV_BETA,
    UPPER_RED1,
    UPPER_RED2,
    V_HSV_ALPHA,
    V_HSV_BETA,
)


# класс предобработки
class CameraPreprocessor:

    def __init__(self, camera_index=0, width=DEFAULT_CAMERA_WIDTH, height=DEFAULT_CAMERA_HEIGHT):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise Exception("Камера не найдена!")

        self.clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE)

    # функция как раз обрабат изображение, яркость + насыщенность подкрутил,
    # также еще с тенями тут полезные фещи накиданы в общем.
    def get_processed_hsv(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

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

        return frame, final_frame

    def release(self):
        self.cap.release()


def get_line_position(frame, center_x):
    height, width = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    roi_y_start = int(height * (1 - ROI_HEIGHT_RATIO))
    roi = mask[roi_y_start:height, :]

    contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return False, 0, 0, mask, roi_y_start

    c = max(contours, key=cv2.contourArea)

    if cv2.contourArea(c) < MIN_CONTOUR_AREA:
        return False, 0, 0, mask, roi_y_start

    M = cv2.moments(c)
    if M["m00"] == 0:
        return False, 0, 0, mask, roi_y_start

    cx = int(M["m10"] / M["m00"])
    error = center_x - cx

    return True, cx, error, mask, roi_y_start


cap = CameraPreprocessor(ROBOT_VIDEO_STREAM_URL)
if __name__ == "__main__":
    while True:
        original_frame, frame = cap.get_processed_hsv()

        # размытие чтобы не было шума визуаольного
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        #  переводим из BGR в HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # две маски для красного цвета
        mask1 = cv2.inRange(hsv, RED_MASK_LOWER1, RED_MASK_UPPER1)
        mask2 = cv2.inRange(hsv, RED_MASK_LOWER2, RED_MASK_UPPER2)

        mask = cv2.bitwise_or(mask1, mask2)

        # 5. убираем белые точки внутри маски и черные снаружи
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, MORPH_KERNEL)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, MORPH_KERNEL)

        red_part = cv2.bitwise_and(frame, frame, mask=mask)

        pure_red = np.full_like(frame, (0, 0, 255), dtype=np.uint8)

        # смешиваем оригинальный красный объект с идеальным красным цветом
        enhanced_red = cv2.addWeighted(red_part, 0.3, pure_red, 0.7, 0)

        # Снова применяем маску, чтобы усиленный красный не вылез за границы объекта
        enhanced_red = cv2.bitwise_and(enhanced_red, enhanced_red, mask=mask)

        # 6. ЧБ версия исходного кадра
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # конвертируем обратно в BGR
        gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        # накладываем цветной красный объект на ЧБ фон
        # ЧБ пиксели там, где маска черная
        background_part = cv2.bitwise_and(gray_bgr, gray_bgr, mask=cv2.bitwise_not(mask))
        result = cv2.add(enhanced_red, background_part)

        # 8. ищем контуры красного объекта, обводим рамкой
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            # фильтр на малые объекты
            if cv2.contourArea(cnt) > RED_CONTOUR_AREA_THRESHOLD:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(result, "Red Object", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Original", frame)
        cv2.imshow("Red Mask", mask)
        cv2.imshow("Result (Red on Grayscale)", result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()