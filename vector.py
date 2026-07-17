import math
import warnings
import numpy as np
from scipy.interpolate import splev
from scipy.integrate import quad, IntegrationWarning
from scipy.optimize import root_scalar

# Подавляем предупреждения о точности интегрирования (не критичны для навигации)
warnings.filterwarnings('ignore', category=IntegrationWarning)
from pid import PIDController
from spline import Spline
from algo import get_wheel_speeds


def evaluate_spline(tck, u):
    """
    Вычисление точки на сплайне через scipy.splev.
    tck: тройка (t, c, k) от splprep
    u: параметр [0..1]
    """
    res = splev(u, tck)
    return float(res[0]), float(res[1])


def get_point_ahead(tck, u, distance=30.0):
    """
    Вычисляет точку на сплайне, находящуюся на заданном расстоянии (distance)
    вперед по дуге от текущего параметра u.
    """
    # 1. Подынтегральная функция: модуль вектора первой производной (ds/du)
    def derivative_magnitude(p):
        dx, dy = splev(p, tck, der=1)
        return np.hypot(dx, dy)

    # 2. Целевая функция для поиска корня
    def objective(u_new):
        length, _ = quad(derivative_magnitude, u, u_new, limit=200)
        return length - distance

    # Определяем конец сплайна (параметр u_max = 1.0 для splprep)
    u_max = 1.0

    # Проверяем, хватает ли оставшейся длины кривой
    total_remaining_length, _ = quad(derivative_magnitude, u, u_max, limit=200)

    if total_remaining_length <= distance:
        # Если до конца траектории меньше distance, возвращаем последнюю точку
        res_x, res_y = splev(u_max, tck)
        return float(res_x), float(res_y), u_max

    # 3. Ищем параметр u_target
    sol = root_scalar(objective, bracket=[u, u_max], method='brentq')
    u_target = sol.root

    # 4. Вычисляем и возвращаем X, Y для найденного параметра
    res_x, res_y = splev(u_target, tck)
    return float(res_x), float(res_y), u_target


def calculate_deviation(start_point, end_point, current_heading=None, max_distance=100.0, max_angle_rad=math.pi / 4):
    """
    Рассчитывает линейное и угловое отклонение между двумя точками,
    а также нормализует их в диапазон [-1.0, 1.0].
    """
    x1, y1 = start_point
    x2, y2 = end_point

    # 1. Вектор смещения по осям
    dx = x2 - x1
    dy = y2 - y1

    # 2. Линейное отклонение (Евклидово расстояние)
    distance = math.hypot(dx, dy)

    # Нормализация дистанции
    norm_dist = distance / max_distance
    norm_dist = max(0.0, min(1.0, norm_dist))

    result = {
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "norm_distance": float(norm_dist)
    }

    # 3. Угловое отклонение
    if current_heading is not None:
        target_angle = math.atan2(dy, dx)
        angle_error = (target_angle - current_heading + math.pi) % (2 * math.pi) - math.pi

        norm_angle = angle_error / max_angle_rad
        norm_angle = max(-1.0, min(1.0, norm_angle))

        result["angle_error_rad"] = angle_error
        result["angle_error_deg"] = math.degrees(angle_error)
        result["norm_angle"] = float(norm_angle)

    return result


class VectorNavigator:
    """
    Навигационный модуль, связывающий Spline, PID и algo.
    
    Принимает точки траектории, строит сплайн, и на каждом шаге
    вычисляет управляющее воздействие (скорости колёс) по текущему
    положению робота.
    """

    def __init__(self, trajectory_points, lookahead_distance=30.0):
        """
        trajectory_points: список точек [(x1,y1), (x2,y2), ...] от Environment
        lookahead_distance: расстояние "взгляда вперёд" по сплайну (в пикселях)
        """
        self.lookahead_distance = lookahead_distance

        # Строим сплайн через точки траектории
        self.update_trajectory(trajectory_points)

        # Текущий параметр на сплайне (0..1)
        self.u = 0.0
        self.finished = False

        # PID и состояние
        self.controller = PIDController()

    def update_trajectory(self, trajectory_points):
        self.spline = Spline()
        xs = np.array([p[0] for p in trajectory_points], dtype=float)
        ys = np.array([p[1] for p in trajectory_points], dtype=float)
        
        # Ensure we have enough points for a cubic spline (k=3 requires at least 4 points)
        if len(xs) < 4:
            self.tck = None
            return
            
        self.spline.splineFromPoints(xs, ys)
        self.tck = self.spline.get_tck()
        self.u = 0.0
        self.finished = False

        # PID и состояние
        self.controller = PIDController()

    def reset(self):
        """Сброс навигатора для нового эпизода (если траектория та же)."""
        self.u = 0.0
        self.finished = False
        self.controller.reset()

    def compute_action(self, robot_x, robot_y, robot_theta):
        """
        Вычисляет действие [vl_norm, vr_norm] для текущего положения робота.
        
        :param robot_x: X-координата робота
        :param robot_y: Y-координата робота
        :param robot_theta: угол поворота робота (рад)
        :return: [left_speed, right_speed] в диапазоне [-1.0, 1.0]
        """
        if self.finished or getattr(self, 'tck', None) is None:
            return [0.0, 0.0]

        # 1. Находим ближайшую точку на сплайне к роботу,
        #    чтобы обновить текущий параметр u
        self._update_u(robot_x, robot_y)

        # 2. Находим точку на lookahead_distance впереди по сплайну
        target_x, target_y, u_ahead = get_point_ahead(
            self.tck, self.u, distance=self.lookahead_distance
        )

        # 3. Проверяем достижение конца
        if u_ahead >= 0.999:
            dist_to_end = math.hypot(target_x - robot_x, target_y - robot_y)
            if dist_to_end < 15.0:
                self.finished = True
                return [0.0, 0.0]

        # 4. Считаем отклонение
        deviation = calculate_deviation(
            (robot_x, robot_y),
            (target_x, target_y),
            current_heading=robot_theta
        )

        # 5. PID-регулятор (принимает ошибку напрямую)
        pid_output = self.controller.get_action(deviation["norm_angle"])

        # 6. Рассчитываем скорости колёс
        vl_norm, vr_norm = get_wheel_speeds(
            pid_output, deviation["angle_error_deg"], found=True
        )

        return [vl_norm, vr_norm]

    def _update_u(self, robot_x, robot_y):
        """
        Обновляет параметр self.u, находя ближайшую точку на сплайне
        к текущей позиции робота. Ищет только вперёд от текущего u.
        """
        best_u = self.u
        best_dist = float('inf')

        # Сканируем вперёд от текущего u с мелким шагом
        search_start = self.u
        search_end = min(self.u + 0.15, 1.0)  # Не ищем слишком далеко вперёд
        num_samples = 30

        for u_test in np.linspace(search_start, search_end, num_samples):
            sx, sy = evaluate_spline(self.tck, u_test)
            d = math.hypot(sx - robot_x, sy - robot_y)
            if d < best_dist:
                best_dist = d
                best_u = u_test

        self.u = best_u

    def is_finished(self):
        """Проверка: достиг ли робот конца сплайна."""
        return self.finished