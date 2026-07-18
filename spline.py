import numpy as np
from scipy.interpolate import splprep, splev

class Spline:
    def __init__(self, smoothing: float = 0.0):
        self.tck = None
        self.u_range = None
        # Добавляем таблицы для мгновенного поиска по расстоянию (оптимизация FPS)
        self.u_fine = np.array([])
        self.arc_lengths = np.array([])
        self.total_length = 0.0
        self.smoothing = smoothing

    def splineFromPoints(self, X: np.ndarray, y: np.ndarray) -> None:
        """Строит сплайн через заданные точки и предрассчитывает длины дуг."""
        if len(X) != len(y) or len(X) < 2:
            raise ValueError("Для сплайна нужны минимум две точки с координатами X и y.")

        # Кубический сплайн требует четыре точки; для короткого участка понижаем степень.
        degree = min(3, len(X) - 1)
        self.tck, self.u_range = splprep([X, y], s=self.smoothing, k=degree)
        self._precompute_arc_lengths()

    def _precompute_arc_lengths(self, resolution: int = 500) -> None:
        """Оптимизация: считаем кумулятивную длину один раз при построении."""
        self.u_fine = np.linspace(0, 1, resolution)
        x_fine, y_fine = splev(self.u_fine, self.tck)
        
        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        segment_lengths = np.hypot(dx, dy)
        
        self.arc_lengths = np.insert(np.cumsum(segment_lengths), 0, 0.0)
        self.total_length = self.arc_lengths[-1]

    def pointsFromSpline(self, num_points: int = 100) -> np.ndarray:
        if self.tck is None:
            raise ValueError("Spline does not exist")
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine = splev(u_fine, self.tck)
        return np.vstack((x_fine, y_fine)).T

    def get_coefficients(self):
        if self.tck is None:
            raise ValueError("Spline does not exist. Call splineFromPoints first.")
        return self.tck[1]

    def get_tck(self):
        if self.tck is None:
            raise ValueError("Spline does not exist. Call splineFromPoints first.")
        return self.tck

    def get_knots(self):
        if self.tck is None:
            raise ValueError("Spline does not exist. Call splineFromPoints first.")
        return self.tck[0]
