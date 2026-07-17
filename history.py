from collections import deque
import numpy as np
from dataclasses import dataclass
from enum import Enum
class ObjectClass(Enum):
    STATIC = 0
    LEFT_RIGHT = 1
    ROUND = 2

@dataclass
class ObjectDetection:
    marker_id: int
    frame: int
    object_class: ObjectClass
    position: np.ndarray
    timestamp: float

@dataclass
class Observation:
    position: np.ndarray
    timestamp: float

class HistoryManager:
    def __init__(self, history_length=30):
        self.history_length = history_length
        self.histories = {}

    def update(self, detections):
        for detection in detections:
            if detection.marker_id not in self.histories:
                self.histories[detection.marker_id] = deque(
                    maxlen=self.history_length
                )
            self.histories[detection.marker_id].append(
                Observation(
                    position=detection.position,
                    timestamp=detection.timestamp
                )
            )
    def get_window(self, id):
        if id in self.histories:
            return list(self.histories[id])
        return []
    
    def has_window(self, id):
        return id in self.histories
    
    def clear_lost_objects(self, current_time):
        """Удаляет объекты, которые не были обнаружены в течение длительного времени."""
        to_remove = []
        for marker_id, history in self.histories.items():
            if history and (current_time - history[-1].timestamp) > 5.0:
                to_remove.append(marker_id)
        for marker_id in to_remove:
            del self.histories[marker_id]