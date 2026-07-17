# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from scipy.interpolate import splprep, splev

class Spline:
    def __init__(self):
        self.tck = None
        self.u_range = None

    def splineFromPoints(self, X: np.ndarray, y: np.ndarray) -> None:
        """Строит сплайн через заданные точки."""
        self.tck, self.u_range = splprep([X, y], s=0)
        
    def pointsFromSpline(self, num_points: int = 100) -> np.ndarray:
        """Возвращает массив точек (x, y) вдоль построенного сплайна."""
        if self.tck is None:
            raise ValueError("Spline does not exist")
            
        u_fine = np.linspace(0, 1, num_points)
        x_fine, y_fine = splev(u_fine, self.tck)
        
        return np.vstack((x_fine, y_fine)).T
    def get_coefficients(self):
        """Возвращает коэффициенты сплайна [c_x, c_y]."""
        if self.tck is None:
            raise ValueError("Spline does not exist. Call splineFromPoints first.")
        
        # Коэффициенты лежат под индексом 1
        return self.tck[1]

    def get_tck(self):
        """Возвращает полную тройку (t, c, k) для scipy.splev."""
        if self.tck is None:
            raise ValueError("Spline does not exist. Call splineFromPoints first.")
        return self.tck

    def get_knots(self):
        """Возвращает узловые точки t."""
        if self.tck is None:
            raise ValueError("Spline does not exist. Call splineFromPoints first.")
        return self.tck[0]
