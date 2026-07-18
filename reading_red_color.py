import cv2
import numpy as np
import time

from logger import log

# ============ НАСТРОЙКИ ============
stream_url = "http://10.136.128.14:5000/video_feed"
ER_INFO = [False, 0, 0]

ROI_HEIGHT_RATIO = 1    #параметр отвечающий за часть экрана которую анализируем

# диапазоны красного меняем если черный цвет нужен
LOWER_RED1 = np.array([0, 100, 60])
UPPER_RED1 = np.array([5, 255, 255])
LOWER_RED2 = np.array([155, 50, 60])
UPPER_RED2 = np.array([180, 255, 255])

MIN_CONTOUR_AREA = 20

def get_line_position(frame, center_x):
    height, width = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, LOWER_RED1, UPPER_RED1)
    mask2 = cv2.inRange(hsv, LOWER_RED2, UPPER_RED2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # вырезаем нижнюю часть кадра
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


def main():
    cap = cv2.VideoCapture(stream_url)
    # Убираем принудительную установку разрешения, берем реальное из кадра!

    if not cap.isOpened():
        log.error("Камера не открылась")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            log.error("Не удалось получить кадр с камеры")
            break

        # УЗНАЕМ РЕАЛЬНЫЙ РАЗМЕР КАДРА
        height, width = frame.shape[:2]
        center_x = width // 2

        # получаем координаты + дополнительные данные для отладки
        found, cx, error, mask, roi_y_start = get_line_position(frame, center_x)

        log.info(f"Line found: {found}, Center X: {cx}, Error: {error}")

        # === БЛОК ОТЛАДКИ ===   (закоментировать если все ок)
        #зона поиска на оригинальном кадре
        #debug_frame = frame.copy()
        #cv2.rectangle(debug_frame, (0, roi_y_start), (width, height), (255, 255, 0), 2)
        #cv2.line(debug_frame, (center_x, 0), (center_x, height), (255, 0, 0), 1) # синяя линия центра

        #if found:
            # рисуем точку центра найденной линии
            #cv2.circle(debug_frame, (cx, roi_y_start + 20), 10, (0, 255, 0), -1)
            #cv2.putText(debug_frame, f"Error: {error}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #else:
            #cv2.putText(debug_frame, "Line NOT found", (10, 30),
                        #cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # 2. отображение маски
        #cv2.imshow('Mask (Red)', mask)

        # 3. кадр с графикой
        #cv2.imshow('Debug View', debug_frame)
        # ==================================

        #print(ER_INFO[2])

        #if cv2.waitKey(10) & 0xFF == ord('q'):
            #break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()