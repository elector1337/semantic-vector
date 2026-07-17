import cv2
import numpy as np
import time
import requests
import threading
import torch

from filter import CameraPreprocessor
from reading_red_color import get_line_position
from algo import get_wheel_speeds
from pid import PIDController
from enum import Enum
from state_chain import StateNode, StateChain, RecoveryController
import numpy as np
from vector import VectorNavigator
from spline import Spline

# ============ НАСТРОЙКИ СЕТИ И РОБОТА ============
ROBOT_URL = "http://198.162.2.2:5000"
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

class Mode(Enum):
    RECORDING = 1
    NAVIGATING = 2


def network_worker():
    """Выделенный поток, который отправляет команды на робота со строгим интервалом"""
    global shared_speed_l, shared_speed_r, network_thread_running

    last_sent_l = 0
    last_sent_r = 0

    while network_thread_running:
        start_time = time.time()
        with speed_lock:
            current_target_l = shared_speed_l
            current_target_r = shared_speed_r

        if current_target_l != last_sent_l or current_target_r != last_sent_r:
            req_url = f"{ROBOT_URL}/move?left={current_target_l}&right={current_target_r}"
            try:
                requests.get(req_url, timeout=0.15)
                last_sent_l = current_target_l
                last_sent_r = current_target_r
            except requests.exceptions.RequestException:
                pass

        elapsed = time.time() - start_time
        sleep_time = max(0.001, COMMAND_SEND_INTERVAL - elapsed)
        time.sleep(sleep_time)


def main():
    global shared_speed_l, shared_speed_r, network_thread_running

    stream_url = f"{ROBOT_URL}/video_feed"
    print("Подключение к камере...")

    filt = CameraPreprocessor(stream_url)

    print("Инициализация ПИД-регулятора...")
    pid_controller = PIDController(Kp=1.0, Ki=0.02, Kd=0.5)
    pid_controller.reset()

    newcameramtx = None
    net_thread = threading.Thread(target=network_worker, daemon=True)
    net_thread.start()

    print(f"Подключение установлено. Режим: {'AI' if USE_RL_AGENT else 'PID'}. (Нажми 'q' для выхода, '1' - Запись, '2' - Навигация, 'r' - Сброс)")

    current_mode = Mode.RECORDING
    chain = StateChain()
    recovery = RecoveryController(chain)
    simulated_odometry = 0.0
    is_recovering = False
    last_line_error = 0.0
    last_time = time.time()
    
    # Dummy init for navigator
    navigator = VectorNavigator([(0,0), (1,0), (2,0), (3,0)])
    
    target_speed_l = 0
    target_speed_r = 0

    try:
        while True:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            _, frame = filt.get_processed_hsv()
            if frame is None:
                print("Ошибка чтения кадра!")
                continue

            if MTX is not None and DIST is not None:
                h, w = frame.shape[:2]
                if newcameramtx is None:
                    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(MTX, DIST, (w, h), 1, (w, h))
                frame = cv2.undistort(frame, MTX, DIST, None, newcameramtx)

            height, width = frame.shape[:2]
            center_x = width // 2

            found, cx, error, mask, roi_y_start = get_line_position(frame, center_x)

            normalized_dx = np.clip(error / (width / 2.0), -1.0, 1.0) if found else 0.0

            if not found:
                if current_mode == Mode.NAVIGATING:
                    if not is_recovering:
                        is_recovering = True
                        recovery.start_recovery(simulated_odometry)
                    
                    # RHC: Строим локальный сплайн каждый кадр
                    recovery.current_memory_idx = chain.find_nearest(simulated_odometry)
                    pts = chain.to_local_spline_points(recovery.current_memory_idx, 30)
                    
                    if len(pts) > 3:
                        pts = np.vstack(([0.0, 0.0], pts))
                        navigator.update_trajectory(pts)
                        # Робот всегда в (0,0) с углом 0 относительно только что построенного сплайна
                        vl, vr = navigator.compute_action(0.0, 0.0, 0.0)
                        target_speed_l = int(vl * MAX_MOTOR_SPEED)
                        target_speed_r = int(vr * MAX_MOTOR_SPEED)
                    else:
                        target_speed_l, target_speed_r = 0, 0
                else:
                    target_speed_l, target_speed_r = 0, 0
                    pid_controller.reset()
            else:
                if current_mode == Mode.NAVIGATING and is_recovering:
                    is_recovering = False
                    recovery.stop_recovery()
                    
                curvature = error - last_line_error
                last_line_error = error
                
                u = pid_controller.get_action(-normalized_dx)
                ls, rs = get_wheel_speeds(u, error, found)
                target_speed_l = int(ls * MAX_MOTOR_SPEED)
                target_speed_r = int(rs * MAX_MOTOR_SPEED)
                
                if current_mode == Mode.RECORDING:
                    node = StateNode(
                        timestamp=current_time,
                        odometry=simulated_odometry,
                        steering_angle=u,
                        line_error=error,
                        line_curvature=curvature,
                        speed=(target_speed_l + target_speed_r) / 2.0
                    )
                    chain.record(node)
                    
            v_current = (target_speed_l + target_speed_r) / 2.0
            simulated_odometry += abs(v_current) * dt

            with speed_lock:
                shared_speed_l = target_speed_l
                shared_speed_r = target_speed_r

            red_part = cv2.bitwise_and(frame, frame, mask=mask)
            pure_red = np.full_like(frame, (0, 0, 255), dtype=np.uint8)
            enhanced_red = cv2.addWeighted(red_part, 0.3, pure_red, 0.7, 0)
            enhanced_red = cv2.bitwise_and(enhanced_red, enhanced_red, mask=mask)

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
            background_part = cv2.bitwise_and(gray_bgr, gray_bgr, mask=cv2.bitwise_not(mask))
            result = cv2.add(enhanced_red, background_part)

            debug_frame = frame.copy()
            cv2.rectangle(debug_frame, (0, roi_y_start), (width, height), (255, 255, 0), 2)
            cv2.line(debug_frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)

            if found:
                cv2.circle(debug_frame, (cx, roi_y_start + 20), 10, (0, 255, 0), -1)
                cv2.putText(debug_frame, f"Error: {error:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(debug_frame, "Line NOT found", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            
            cv2.putText(debug_frame, f"Mode: {current_mode.name} {'(RECOVERY)' if is_recovering else ''}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            cv2.putText(debug_frame, f"Odom: {simulated_odometry:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            cv2.imshow('Mask (Red)', mask)
            cv2.imshow('Debug View', debug_frame)

            key = cv2.waitKey(10) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('1'):
                current_mode = Mode.RECORDING
                is_recovering = False
                simulated_odometry = 0.0
                chain = StateChain()
                recovery = RecoveryController(chain)
                print("Режим: RECORDING (Память очищена)")
            elif key == ord('2'):
                current_mode = Mode.NAVIGATING
                is_recovering = False
                simulated_odometry = 0.0
                print("Режим: NAVIGATING")
            elif key == ord('r'):
                is_recovering = False
                simulated_odometry = 0.0
                if current_mode == Mode.RECORDING:
                    chain = StateChain()
                    recovery = RecoveryController(chain)
                print("Сброс одометрии!")

    finally:
        print("Остановка робота и закрытие ресурсов...")
        network_thread_running = False
        try:
            requests.get(f"{ROBOT_URL}/stop", timeout=0.5)
        except requests.exceptions.RequestException:
            pass
        filt.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
