import numpy as np
from constants import PID_KP, PID_KI, PID_KD, PID_DT, PID_MAX_SPEED

class PIDController:
    def __init__(self, Kp=PID_KP, Ki=PID_KI, Kd=PID_KD, dt=PID_DT, max_speed=PID_MAX_SPEED):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.max_speed = max_speed

        self.integral = 0.0
        self.last_error = 0.0



    def reset(self):
        """Сброс состояния при начале нового эпизода"""
        self.integral = 0.0
        self.last_error = 0.0


    def get_action(self, error):
        # Читаем ошибку с камеры

        # Вычисляем ПИД
        self.integral += error * self.dt
        derivative = (error - self.last_error) / self.dt
        control_signal = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.last_error = error

        # Нормализуем желаемое действие в диапазон [-1, 1]
        desired_action = np.clip(control_signal / self.max_speed, -1.0, 1.0)



        return float(desired_action)