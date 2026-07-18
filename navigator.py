import math
import warnings
import numpy as np
from scipy.interpolate import splev
from scipy.optimize import minimize_scalar
from scipy.integrate import IntegrationWarning

warnings.filterwarnings('ignore', category=IntegrationWarning)

from pid import PIDController
from spline import Spline
from algo import get_wheel_speeds
from constants import MIN_LOCAL_PATH_POINTS
import logging
logger = logging.getLogger(__name__)

def evaluate_spline(tck, u):
    res = splev(u, tck)
    return float(res[0]), float(res[1])

def get_point_ahead_optimized(spline_obj, u, distance=30.0):
    """
    ОПТИМИЗИРОВАННАЯ ВЕРСИЯ: Находит точку на расстоянии distance.
    Работает за O(log N) благодаря предрасчитанной таблице, а не тяжелому quad.
    """
    # 1. Текущая длина дуги для данного u (интерполяция)
    current_dist = np.interp(u, spline_obj.u_fine, spline_obj.arc_lengths)
    target_dist = current_dist + distance

    # 2. Если расстояние больше оставшейся длины, возвращаем конец
    if target_dist >= spline_obj.total_length:
        res_x, res_y = splev(1.0, spline_obj.tck)
        return float(res_x), float(res_y), 1.0

    # 3. Бинарный поиск нужного индекса в таблице длин
    idx = np.searchsorted(spline_obj.arc_lengths, target_dist)
    
    # 4. Линейная интерполяция параметра u между двумя узлами для точности
    if idx == 0:
        u_target = 0.0
    else:
        seg_len = spline_obj.arc_lengths[idx] - spline_obj.arc_lengths[idx - 1]
        if seg_len == 0:
            u_target = spline_obj.u_fine[idx]
        else:
            t = (target_dist - spline_obj.arc_lengths[idx - 1]) / seg_len
            u_target = spline_obj.u_fine[idx - 1] + t * (spline_obj.u_fine[idx] - spline_obj.u_fine[idx - 1])

    res_x, res_y = splev(u_target, spline_obj.tck)
    return float(res_x), float(res_y), float(u_target)

def calculate_deviation(start_point, end_point, current_heading=None, max_distance=100.0, max_angle_rad=math.pi / 4):
    x1, y1 = start_point
    x2, y2 = end_point
    
    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)
    
    norm_dist = max(0.0, min(1.0, distance / max_distance))
    
    result = {
        "dx": dx, "dy": dy, "distance": distance, "norm_distance": float(norm_dist)
    }
    
    if current_heading is not None:
        target_angle = math.atan2(dy, dx)
        angle_error = (target_angle - current_heading + math.pi) % (2 * math.pi) - math.pi
        norm_angle = max(-1.0, min(1.0, angle_error / max_angle_rad))
        
        result["angle_error_rad"] = angle_error
        result["angle_error_deg"] = math.degrees(angle_error)
        result["norm_angle"] = float(norm_angle)
        
    return result

class VectorNavigator:
    def __init__(self, trajectory_points, lookahead_distance=30.0):
        self.lookahead_distance = lookahead_distance
        self.spline = Spline()
        
        xs = np.array([p[0] for p in trajectory_points], dtype=float)
        ys = np.array([p[1] for p in trajectory_points], dtype=float)
        self.spline.splineFromPoints(xs, ys)
        self.tck = self.spline.get_tck()
        
        self.u = 0.0
        self.finished = False
        self.controller = PIDController()

    def reset(self):
        self.u = 0.0
        self.finished = False
        self.controller.reset()

    def compute_action(self, robot_x, robot_y, robot_theta):
        if self.finished:
            return [0.0, 0.0]

        # 1. Находим ближайшую точку (оптимизировано)
        self._update_u(robot_x, robot_y)

        # 2. Находим точку вперед (теперь работает мгновенно)
        target_x, target_y, u_ahead = get_point_ahead_optimized(
            self.spline, self.u, distance=self.lookahead_distance
        )

        # 3. Проверяем достижение конца
        if u_ahead >= 0.999:
            dist_to_end = math.hypot(target_x - robot_x, target_y - robot_y)
            if dist_to_end < 15.0:
                self.finished = True
                return [0.0, 0.0]

        # 4. Считаем отклонение
        deviation = calculate_deviation(
            (robot_x, robot_y), (target_x, target_y), current_heading=robot_theta
        )

        # 5. PID и скорости
        pid_output = self.controller.get_action(deviation["norm_angle"])
        vl_norm, vr_norm = get_wheel_speeds(pid_output, deviation["angle_error_deg"], found=True)
        
        return [vl_norm, vr_norm]

    def _update_u(self, robot_x, robot_y):
        """
        ОПТИМИЗИРОВАНО: Вместо перебора 30 точек используем bounded минимизацию.
        Это математически точно находит минимум расстояния и работает быстрее.
        """
        search_start = self.u
        search_end = min(self.u + 0.15, 1.0)
        
        if search_start >= search_end:
            self.u = 1.0
            return

        def distance_squared(u_val):
            sx, sy = evaluate_spline(self.tck, u_val)
            return (sx - robot_x)**2 + (sy - robot_y)**2

        res = minimize_scalar(
            distance_squared, 
            bounds=(search_start, search_end), 
            method='bounded'
        )
        self.u = float(res.x)

    def is_finished(self):
        return self.finished


class LocalSplineNavigator:
    """Строит локальный сплайн по линии текущего кадра и выбирает look-ahead точку."""

    def __init__(self, lookahead_distance, smoothing):
        self.lookahead_distance = lookahead_distance
        self.smoothing = smoothing

    def get_angle_error(self, centerline, robot_point):
        """Возвращает ошибку угла [-1; 1] и look-ahead точку либо (None, None)."""
        if len(centerline) < MIN_LOCAL_PATH_POINTS:
            return None, None

        robot = np.asarray(robot_point, dtype=float)
        points = np.asarray(centerline, dtype=float)
        # Робот — начало локальной траектории. Удаляем совпадающие с ним точки,
        # иначе splprep получает повторяющиеся параметры.
        points = points[np.linalg.norm(points - robot, axis=1) > 1.0]
        if len(points) < MIN_LOCAL_PATH_POINTS:
            return None, None

        trajectory = np.vstack((robot, points))
        spline = Spline(smoothing=self.smoothing)
        try:
            spline.splineFromPoints(trajectory[:, 0], trajectory[:, 1])
        except (TypeError, ValueError):
            return None, None

        target_x, target_y, _ = get_point_ahead_optimized(
            spline, u=0.0, distance=self.lookahead_distance
        )

        # В изображении ось Y направлена вниз. Направление робота — вверх кадра.
        lateral = target_x - robot[0]
        forward = robot[1] - target_y
        if forward <= 0:
            return None, None

        angle_error = math.atan2(lateral, forward)
        normalized_angle = float(np.clip(angle_error / (math.pi / 2), -1.0, 1.0))
        return normalized_angle, (int(round(target_x)), int(round(target_y)))
