import cv2
import numpy as np
import time
import requests
import threading
import torch

from filter import CameraPreprocessor
from filter import get_line_position
from algo import get_wheel_speeds
from pid import PIDController
from logging import log

# ============ НАСТРОЙКИ СЕТИ И РОБОТА ============
ROBOT_URL = "http://10.136.128.14:5000"
MAX_MOTOR_SPEED = 200
USE_RL_AGENT = False  # Переключатель: False = работает PID, True = работает Нейросеть

# Ограничение частоты отправки команд на сервер (0.1 сек = 10 FPS)
COMMAND_SEND_INTERVAL = 0.1

# === МАТРИЦЫ КАЛИБРОВКИ ===
MTX = None
DIST = None

# === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ ПОТОКА УПРАВЛЕНИЯ ===
speed_lock = threading.Lock()
shared_speed_l = 0
shared_speed_r = 0
network_thread_running = True


def network_worker():
    """Выделенный поток, который отправляет команды на робота со строгим интервалом"""
    global shared_speed_l, shared_speed_r, network_thread_running

    # Храним состояние скоростей локально внутри потока, чтобы знать, что изменилось
    last_sent_l = 0
    last_sent_r = 0

    while network_thread_running:
        start_time = time.time()

        # Безопасно забираем текущие целевые скорости
        with speed_lock:
            current_target_l = shared_speed_l
            current_target_r = shared_speed_r

        # Отправляем только если скорости изменились ИЛИ для поддержания связи (watchdog)
        # Если нужно отправлять ВСЕГДА (каждые 100мс), можно убрать условие с "if"
        if current_target_l != last_sent_l or current_target_r != last_sent_r:
            req_url = f"{ROBOT_URL}/move?left={current_target_l}&right={current_target_r}"
            try:
                requests.get(req_url, timeout=0.15)
                last_sent_l = current_target_l
                last_sent_r = current_target_r
            except requests.exceptions.RequestException:
                pass  # Сервер временно недоступен — игнорируем ошибку

        # Рассчитываем точное время сна, учитывая время выполнения самого запроса
        elapsed = time.time() - start_time
        sleep_time = max(0.001, COMMAND_SEND_INTERVAL - elapsed)
        time.sleep(sleep_time)


# ============ ГЛАВНАЯ ФУНКЦИЯ ============
def main():
    global shared_speed_l, shared_speed_r, network_thread_running

    stream_url = f"{ROBOT_URL}/video_feed"

    filt = CameraPreprocessor(stream_url)

    # === ИНИЦИАЛИЗАЦИЯ УПРАВЛЕНИЯ ===
    pid_controller = PIDController(Kp=1.0, Ki=0.02, Kd=0.5)
    pid_controller.reset()

    newcameramtx = None

    # Запуск выделенного сетевого потока
    net_thread = threading.Thread(target=network_worker, daemon=True)
    net_thread.start()

    log.info(f"Подключение установлено. Режим: {'AI' if USE_RL_AGENT else 'PID'}. (Нажми 'q' для выхода)")

    try:
        while True:
            _, frame = filt.get_processed_hsv()
            if frame is None:
                log.error("Ошибка чтения кадра!")
                continue

            # 1. Выпрямление кадра (если заданы матрицы)
            if MTX is not None and DIST is not None:
                h, w = frame.shape[:2]
                if newcameramtx is None:
                    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 1, (w, h))
                frame = cv2.undistort(frame, MTX, DIST, None, newcameramtx)

            height, width = frame.shape[:2]
            center_x = width // 2
            blurred = cv2.GaussianBlur(frame, (5, 5), 0)

            # Переводим из BGR в HSV
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 120, 100])
            upper_red1 = np.array([5, 255, 255])
            lower_red2 = np.array([175, 120, 100])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)

            # Убираем шумы
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            red_part = cv2.bitwise_and(frame, frame, mask=mask)
            pure_red = np.full_like(frame, (0, 0, 255), dtype=np.uint8)

            enhanced_red = cv2.addWeighted(red_part, 0.3, pure_red, 0.7, 0)
            enhanced_red = cv2.bitwise_and(enhanced_red, enhanced_red, mask=mask)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

            background_part = cv2.bitwise_and(gray_bgr, gray_bgr, mask=cv2.bitwise_not(mask))
            result = cv2.add(enhanced_red, background_part)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) > 500:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(result, "Red Object", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # 2. ПОЛУЧЕНИЕ ДАННЫХ ИЗ МОДУЛЯ ЗРЕНИЯ (поиск линии)
            found, cx, error, mask, roi_y_start = get_line_position(result, center_x)

            # Нормализация ошибки
            normalized_dx = np.clip(error / (width / 2.0), -1.0, 1.0) if found else 0.0

            # 3. ВЫЧИСЛЕНИЕ УПРАВЛЯЮЩЕГО СИГНАЛА
            if not found:
                target_speed_l, target_speed_r = 0, 0
                pid_controller.reset()
            else:
                u = pid_controller.get_action(-normalized_dx)
                ls, rs = get_wheel_speeds(u, error, found)

                target_speed_l = int(ls * MAX_MOTOR_SPEED)
                target_speed_r = int(rs * MAX_MOTOR_SPEED)

            # ВМЕСТО ОТПРАВКИ: Безопасно обновляем скорости для сетевого потока
            with speed_lock:
                shared_speed_l = target_speed_l
                shared_speed_r = target_speed_r

            # 5. ОТРИСОВКА И ДЕБАГ
            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (0, roi_y_start), (width, height), (255, 255, 0), 2)
            cv2.line(debug_frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)

            if found:
                cv2.circle(debug_frame, (cx, roi_y_start + 20), 10, (0, 255, 0), -1)
                cv2.putText(debug_frame, f"Error: {error}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(debug_frame, "Line NOT found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow('Mask (Red)', mask)
            cv2.imshow('Debug View', debug_frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    finally:
        # Сигнализируем сетевому потоку о завершении работы
        log.info("Завершение работы.")
        network_thread_running = False

        # Финальный стоп отправляем синхронно (без спавна потока, так как всё закрывается)
        try:
            requests.get(f"{ROBOT_URL}/stop", timeout=0.5)
        except requests.exceptions.RequestException:
            pass

        filt.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()