import cv2
import time
import threading
import serial

from filter import FrameProcessor
from camera import Camera
from navigator import LocalSplineNavigator
from algo import get_wheel_speeds
from pid import PIDController
from constants import (
    BAUD_RATE,
    COMMAND_SEND_INTERVAL,
    MAX_MOTOR_SPEED,
    LOCAL_LOOKAHEAD_DISTANCE,
    LOCAL_SPLINE_SMOOTHING,
    ROBOT_CONTROL_KD,
    ROBOT_CONTROL_KI,
    ROBOT_CONTROL_KP,
    SERIAL_PORT,
)
import logging
logger = logging.getLogger(__name__)

# === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ ПОТОКА УПРАВЛЕНИЯ ===
speed_lock = threading.Lock()
shared_speed_l = 0
shared_speed_r = 0
serial_thread_running = True
ser = None

# Пытаемся открыть Serial-порт при старте
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"[SERIAL] Подключено к порту {SERIAL_PORT} на скорости {BAUD_RATE}")
except serial.SerialException as e:
    print(f"[SERIAL ОШИБКА] Не удалось открыть порт {SERIAL_PORT}: {e}")


def serial_worker():
    global shared_speed_l, shared_speed_r, serial_thread_running, ser

    last_sent_l = 0
    last_sent_r = 0

    while serial_thread_running:
        start_time = time.time()

        # Безопасно забираем текущие целевые скорости
        with speed_lock:
            current_target_l = shared_speed_l
            current_target_r = shared_speed_r

        # Отправляем только если скорости изменились
        if current_target_l != last_sent_l or current_target_r != last_sent_r:
            if ser and ser.is_open:
                # Формируем строку вида "50,-50\n"
                command = f"{current_target_l},{current_target_r}\n"
                try:
                    ser.write(command.encode('utf-8'))
                    last_sent_l = current_target_l
                    last_sent_r = current_target_r
                except Exception as e:
                    print(f"[SERIAL ОШИБКА] Сбой записи: {e}")

        # Рассчитываем точное время сна
        elapsed = time.time() - start_time
        sleep_time = max(0.001, COMMAND_SEND_INTERVAL - elapsed)
        time.sleep(sleep_time)


# ============ ГЛАВНАЯ ФУНКЦИЯ ============
def main(camera=None):
    global shared_speed_l, shared_speed_r, serial_thread_running, ser

    owns_camera = camera is None
    if owns_camera:
        camera = Camera()
        if not camera.capture(1):
            raise RuntimeError("Камера не найдена!")

    frame_processor = FrameProcessor(camera)
    navigator = LocalSplineNavigator(
        lookahead_distance=LOCAL_LOOKAHEAD_DISTANCE,
        smoothing=LOCAL_SPLINE_SMOOTHING,
    )
    pid_controller = PIDController(Kp=ROBOT_CONTROL_KP, Ki=ROBOT_CONTROL_KI, Kd=ROBOT_CONTROL_KD)
    s_thread = threading.Thread(target=serial_worker, daemon=True)

    pid_controller.reset()
    s_thread.start()

    try:
        while True:
            frame, centerline = frame_processor.get_local_path()
            if frame is None:
                logger.warning("ошибка чтения файла")
                continue

            robot_point = (frame.shape[1] // 2, frame.shape[0] - 1)
            angle_error, target_point = navigator.get_angle_error(centerline, robot_point)
            found = angle_error is not None

            if not found:
                target_speed_l, target_speed_r = 0, 0
                pid_controller.reset()
            else:
                cv2.circle(frame, target_point, 6, (0, 255, 255), -1)
                u = pid_controller.get_action(-angle_error)
                ls, rs = get_wheel_speeds(u, angle_error, found)

                target_speed_l = int(ls * MAX_MOTOR_SPEED)
                target_speed_r = int(rs * MAX_MOTOR_SPEED)

            with speed_lock:
                shared_speed_l = target_speed_l
                shared_speed_r = target_speed_r

            time.sleep(0.01)

    except KeyboardInterrupt:
       logger.info("программа остановлена пользоавтелем")

    finally:
        logger.info("Остановка робота и закрытие ресурсов...")
        # Сигнализируем Serial-потоку о завершении
        serial_thread_running = False

        # Экстренный стоп: пишем нули напрямую в порт
        if ser and ser.is_open:
            try:
                ser.write(b"0,0\n")
                ser.close()
                logger.info("[SERIAL] Порт закрыт.")
            except Exception as e:
                logger.info(f"Ошибка при закрытии порта: {e}")

        frame_processor.release()
        if owns_camera:
            camera.release()


if __name__ == "__main__":
    main()
