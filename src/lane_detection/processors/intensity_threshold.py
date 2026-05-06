from __future__ import annotations

import numpy as np

from lane_detection.processors.base import Processor
from lane_detection.utils.thresholding import kapur_threshold

class IntensityThresholdStage(Processor):
    """Keep points whose intensity exceeds the per-window threshold.

    Parameters
    ----------
    method : str, either `kapur` or `percentile`
    offset : float
        Subtracted from each window's threshold (so more points are kept).
    min_threshold : float
        Lower clamp for the threshold after applying ``offset``.
    percentile : float
        Top percentile of points to keep (e.g. 5.0 → top 5%)
    """

    def __init__(
        self,
        method: str = "kapur",
        offset: float = 0.0,
        min_threshold: float = 0.0,
        percentile: float = 5.0,
    ):
        super().__init__()
        self.method = method
        self.offset = float(offset)
        self.min_threshold = float(min_threshold)
        self.percentile = float(percentile)

    def process_window(self, indices: np.ndarray, context) -> np.ndarray:
        intensities = np.asarray(context.las.intensity)
        window_int = intensities[indices]

        if window_int.size == 0:
            return np.zeros(0, dtype=bool)

        if self.method == "kapur":
            theta = kapur_threshold(window_int)

            if theta is None:
                return np.ones(indices.size, dtype=bool)

            threshold = max(theta - self.offset, self.min_threshold)
            return window_int >= threshold

        elif self.method == "percentile":
            perc_value = 100.0 - self.percentile
            theta = np.percentile(window_int, perc_value)

            threshold = max(theta, self.min_threshold)
            return window_int >= threshold

        else:
            raise ValueError(f"Unknown method: {self.method}")