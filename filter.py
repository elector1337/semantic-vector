import cv2
import numpy as np
from logger import log


# класс предобработки
class CameraPreprocessor:

    def __init__(self, camera_index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        if not self.cap.isOpened():
            raise Exception("Камера не найдена!")

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # функция как раз обрабат изображение, яркость + насыщенность подкрутил,
    # также еще с тенями тут полезные фещи накиданы в общем.
    def get_processed_hsv(self):
        ret, frame = self.cap.read()
        if not ret:
            return None, None

        filtered = cv2.bilateralFilter(frame, d=5, sigmaColor=5, sigmaSpace=5)

        lab = cv2.cvtColor(filtered, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_clahe = self.clahe.apply(l)

        lab_clahe = cv2.merge((l_clahe, a, b))
        balanced_bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

        hsv = cv2.cvtColor(balanced_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        s_enhanced = cv2.convertScaleAbs(s, alpha=0.95, beta=-3)
        v_enhanced = cv2.convertScaleAbs(v, alpha=1.1, beta=0)

        final_hsv = cv2.merge((h, s_enhanced, v_enhanced))

        final_frame = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

        return frame, final_frame

    def release(self):
        self.cap.release()


cap = CameraPreprocessor("http://10.136.128.14:5000/video_feed")
if __name__ == "__main__":
    while True:
        original_frame, frame = cap.get_processed_hsv()

        # размытие чтобы не было шума визуаольного
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)

        #  переводим из BGR в HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower_red1 = np.array([0, 120, 100])
        upper_red1 = np.array([5, 255, 255])

        lower_red2 = np.array([175, 120, 100])
        upper_red2 = np.array([180, 255, 255])
        # две маски для красного цвета
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)

        # 5. убираем белые точки внутри маски и черные снаружи
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

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
            if cv2.contourArea(cnt) > 500:
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