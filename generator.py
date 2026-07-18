from history import HistoryManager, ObjectClass, ObjectDetection
import numpy as np
from dataclasses import dataclass
import pandas as pd
from pathlib import Path
from constants import TRACKS_DIR, DATASET_INPUT_WINDOW, DATASET_OUTPUT_WINDOW, DATASET_FEATURES


tracks_dir = TRACKS_DIR
tracks_dir.mkdir(exist_ok=True)

@dataclass
class TrackPoint:
    timestamp: float
    x: float
    z: float
    marker_id: int
    object_class: ObjectClass
    
import pandas as pd
import numpy as np

class PointsStorage:
    def __init__(self):
        self.rows = []
        

    def append(self, detection):
        self.rows.append({
            "frame": detection.frame,
            "timestamp": detection.timestamp,
            "marker_id": detection.marker_id,
            "object_class": detection.object_class.name,
            "x": detection.position[0],
            "z": detection.position[1],
        })

    @property
    def df(self):
        return pd.DataFrame(self.rows)

    def save(self):
        tracks_dir = Path("tracks")
        tracks_dir.mkdir(exist_ok=True)
        track_number = len(list(tracks_dir.glob("track*.parquet")))
        filename = tracks_dir / f"track{track_number}.parquet"
        self.df.to_parquet(filename, index=False)

    def save_json(self, filename="tracks.json"):
        self.df.to_json(filename, orient="records", indent=4)

class DatasetGenerator:
    def __init__(
        self,
        input_window=DATASET_INPUT_WINDOW,
        output_window=DATASET_OUTPUT_WINDOW,
        features=DATASET_FEATURES,
    ):

        self.input_window = input_window
        self.output_window = output_window
        self.features = list(features)

    def generate(self, dataframe: pd.DataFrame):
        X = []
        Y = []
        # сортируем по времени
        dataframe = dataframe.sort_values(
            ["marker_id", "frame"]
        )
        for marker_id, track in dataframe.groupby("marker_id"):
            points = track[self.features].to_numpy(dtype=np.float32)
            total = len(points)
            if total < self.input_window + self.output_window:
                continue
            for start in range(
                total - self.input_window - self.output_window + 1
            ):
                x = points[
                    start :
                    start + self.input_window
                ]
                y = points[
                    start + self.input_window :
                    start + self.input_window + self.output_window
                ]
                X.append(x)
                Y.append(y)

        X = np.asarray(X, dtype=np.float32)
        Y = np.asarray(Y, dtype=np.float32)

        return X, Y

    def save(
        self,
        filename,
        X,
        Y
    ):
        np.savez(
            filename,
            X=X,
            Y=Y
        )