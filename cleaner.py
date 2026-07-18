from pathlib import Path

import numpy as np
import pandas as pd
import logging
logger = logging.getLogger(__name__)


class TrajectoryCleaner:
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe.copy()

    def remove_short_tracks(self, min_length: int = 30):
        """
        Удаляет слишком короткие треки.
        """
        if self.df.empty:
            return self
        counts = self.df.groupby("track_id").size()
        valid_tracks = counts[counts >= min_length].index
        self.df = self.df[self.df["track_id"].isin(valid_tracks)]
        self.df = self.df.reset_index(drop=True)

        return self

    def split_large_jumps(
        self,
        max_distance: float = 25.0,
        max_gap: int = 3,
    ):
        """
        Разбивает трек, если между двумя соседними точками
        слишком большой скачок или большой пропуск кадров.
        """
        if self.df.empty:
            return self

        new_tracks = []
        next_track = 0
        for _, track in self.df.groupby("marker_id"):
            track = track.sort_values("frame").copy()
            current_track = next_track
            ids = []
            prev_x = None
            prev_y = None
            prev_frame = None
            for _, row in track.iterrows():
                if prev_x is not None:
                    dist = np.hypot(
                        row.x - prev_x,
                        row.y - prev_y
                    )
                    gap = row.frame - prev_frame

                    if dist > max_distance or gap > max_gap:
                        current_track += 1

                ids.append(current_track)

                prev_x = row.x
                prev_y = row.y
                prev_frame = row.frame

            track["track_id"] = ids
            next_track = current_track + 1
            new_tracks.append(track)

        self.df = pd.concat(new_tracks, ignore_index=True)
        return self

    def remove_outliers(
        self,
        window: int = 5,
        threshold: float = 2.5,
    ):
        """
        Удаляет одиночные выбросы.
        """
        if self.df.empty:
            return self

        cleaned = []
        for _, track in self.df.groupby(["marker_id", "track_id"]):
            track = track.copy()
            x_med = (
                track["x"]
                .rolling(window, center=True)
                .median()
            )

            y_med = (
                track["y"]
                .rolling(window, center=True)
                .median()
            )

            distance = np.sqrt(
                (track["x"] - x_med) ** 2
                + (track["y"] - y_med) ** 2
            )

            mask = (
                distance.isna()
                | (distance < threshold)
            )

            cleaned.append(track[mask])

        self.df = pd.concat(cleaned, ignore_index=True)

        return self

    def trim_stationary_segments(
        self,
        max_stationary: int = 15,
        epsilon: float = 0.5,
    ):
        """
        Ограничивает длину стоянки.
        """
        if self.df.empty:
            return self

        result = []
        for _, track in self.df.groupby(["marker_id", "track_id"]):
            rows = []
            stationary = 0
            prev = None
            for _, row in track.iterrows():

                if prev is None:
                    rows.append(row)
                    prev = row
                    continue

                dist = np.hypot(
                    row.x - prev.x,
                    row.y - prev.y
                )

                if dist < epsilon:
                    stationary += 1
                else:
                    stationary = 0

                if stationary <= max_stationary:
                    rows.append(row)

                prev = row

            result.append(pd.DataFrame(rows))
        self.df = pd.concat(result, ignore_index=True)

        return self

    def smooth(self, window: int = 5):
        """
        Простое сглаживание скользящим средним.
        """
        if self.df.empty:
            return self

        result = []
        for _, track in self.df.groupby(["marker_id", "track_id"]):

            track = track.copy()

            track["x"] = (
                track["x"]
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

            track["y"] = (
                track["y"]
                .rolling(window, center=True, min_periods=1)
                .mean()
            )

            result.append(track)

        self.df = pd.concat(result, ignore_index=True)

        return self

    def full_clean(
        self,
        min_length=30,
        max_distance=25,
        max_gap=3,
        outlier_window=5,
        outlier_threshold=2.5,
        max_stationary=15,
        stationary_eps=0.5,
        smooth_window=5,
    ):

        return (
            self
            .split_large_jumps(
                max_distance=max_distance,
                max_gap=max_gap,
            )
            .remove_short_tracks(
                min_length=min_length
            )
            .remove_outliers(
                window=outlier_window,
                threshold=outlier_threshold,
            )
            .trim_stationary_segments(
                max_stationary=max_stationary,
                epsilon=stationary_eps,
            )
            .smooth(
                window=smooth_window
            )
        )
    def save(self):
        tracks_dir = Path("tracks/clean")
        tracks_dir.mkdir(exist_ok=True)
        track_number = len(list(tracks_dir.glob("track*.parquet")))
        filename = tracks_dir / f"track{track_number:03}.parquet"
        self.df.to_parquet(filename, index=False)
        logger.debug("clean_tracks.parquet saved")

        return self