import math
import numpy as np
from dataclasses import dataclass
from typing import List

@dataclass
class StateNode:
    """
    Узел состояния (StateNode) хранит информацию о роботе в конкретный момент времени.
    Это своеобразный "снимок" того, что робот видел и делал.
    """
    timestamp: float        # Время записи
    odometry: float         # Пройденная дистанция (одометрия) с момента старта
    steering_angle: float   # Угол руления (команда, отправленная на моторы)
    line_error: float       # Ошибка отклонения от линии (расстояние до центра линии)
    line_curvature: float   # Кривизна линии (производная от ошибки)
    speed: float            # Линейная скорость робота
    
class StateChain:
    """
    Цепочка состояний (StateChain) - это "память" робота.
    Она хранит хронологическую последовательность всех узлов состояния (StateNode),
    формируя пройденный маршрут в виде топологической карты без глобальных координат.
    """
    def __init__(self):
        self.nodes: List[StateNode] = []
        
    def record(self, node: StateNode):
        """Добавляет новый узел в память."""
        self.nodes.append(node)
        
    def find_nearest(self, odometry: float, window_size: int = 50) -> int:
        """
        Находит в памяти индекс узла, одометрия которого наиболее близка к заданной.
        Это позволяет роботу понять, в какой точке записанного маршрута он сейчас находится.
        """
        if not self.nodes:
            return -1
            
        # Простой подход: ищем узел с минимальной разницей в пройденном расстоянии
        best_idx = -1
        best_score = float('inf')
        
        for i, node in enumerate(self.nodes):
            score = abs(node.odometry - odometry)
                
            if score < best_score:
                best_score = score
                best_idx = i
                
        return best_idx
        
    def get_lookahead(self, start_idx: int, n: int) -> List[StateNode]:
        """Возвращает "окно" из следующих n узлов, начиная с заданного индекса."""
        if start_idx < 0 or start_idx >= len(self.nodes):
            return []
        end_idx = min(len(self.nodes), start_idx + n)
        return self.nodes[start_idx:end_idx]
        
    def to_local_spline_points(self, start_idx: int, window: int) -> np.ndarray:
        """
        Главная математическая магия: берет записанные углы руления и расстояния,
        и "разворачивает" их в массив (X, Y) точек перед роботом.
        
        Сгенерированные точки находятся в локальной системе координат,
        где сам робот всегда находится в центре (0, 0) и смотрит прямо (угол 0).
        """
        nodes = self.get_lookahead(start_idx, window)
        if not nodes:
            return np.array([])
            
        points = []
        x, y, theta = 0.0, 0.0, 0.0
        
        for i in range(len(nodes) - 1):
            curr = nodes[i]
            next_node = nodes[i+1]
            
            # ds - разница в расстоянии между текущим и следующим узлом
            ds = next_node.odometry - curr.odometry
            
            # Приближенное вычисление изменения угла (dtheta) на основе записанной команды руля.
            # Коэффициент 0.05 подбирается так, чтобы команда руля (например, ПИД-выход)
            # адекватно переводилась в реальный радиус поворота.
            dtheta = curr.steering_angle * ds * 0.05 
            
            # Интегрируем локальные X и Y
            x += ds * math.cos(theta)
            y += ds * math.sin(theta)
            theta += dtheta
            
            points.append([x, y])
            
        return np.array(points)

class RecoveryController:
    """
    Контроллер восстановления (RecoveryController) отвечает за слепой проезд
    при потере линии. Сейчас он используется для простого воспроизведения угла
    (старый метод), но его индекс (current_memory_idx) также служит стартовой
    точкой для построения локальных сплайнов (новый метод RHC).
    """
    def __init__(self, chain: StateChain):
        self.chain = chain
        self.active = False
        self.current_memory_idx = -1
        
    def start_recovery(self, last_known_odometry: float):
        """Активирует режим восстановления, находя текущую позицию в памяти."""
        self.active = True
        self.current_memory_idx = self.chain.find_nearest(last_known_odometry)
        
    def get_recovery_steering(self, current_odometry: float) -> float:
        """
        Возвращает сохраненный угол руления для текущей одометрии.
        (Оставлен для совместимости, хотя теперь мы предпочитаем строить локальный сплайн)
        """
        if not self.active or self.current_memory_idx == -1:
            return 0.0
            
        # Ищем следующий узел, пока одометрия узла не превысит текущую одометрию робота
        while self.current_memory_idx < len(self.chain.nodes) - 1:
            if self.chain.nodes[self.current_memory_idx].odometry >= current_odometry:
                return self.chain.nodes[self.current_memory_idx].steering_angle
            self.current_memory_idx += 1
            
        return 0.0
        
    def stop_recovery(self):
        """Отключает режим восстановления."""
        self.active = False
