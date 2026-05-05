"""Per-window intensity thresholding using Kapur's maximum entropy method.

For each window (built by :class:`Processor` from bins + window params),
Kapur's max-entropy threshold is computed from the intensity histogram and
points with intensity above ``theta_max - offset`` (clamped to ``min_threshold``)
are kept.
"""
from __future__ import annotations

import numpy as np

from lane_detection.processors.base import Processor
from lane_detection.utils.logger import create_logger
from lane_detection.utils.thresholding import kapur_threshold

class IntensityThresholdStage(Processor):
    """Keep points whose intensity exceeds the per-window Kapur threshold.

    Parameters
    ----------
    offset : float, default 20.0
        Subtracted from each window's threshold (so more points are kept).
    min_threshold : float, default 20.0
        Lower clamp for the threshold after applying ``offset``.
    """

    def __init__(
        self,
        offset: float = 20.0,
        min_threshold: float = 20.0,
    ):
        
        super().__init__()
        self.offset = float(offset)
        self.min_threshold = float(min_threshold)

    def process_window(self, indices: np.ndarray, context) -> np.ndarray:
        intensities = np.asarray(context.las.intensity)
        window_int = intensities[indices]

        theta = kapur_threshold(window_int)
        if theta is None:
            return np.ones(indices.size, dtype=bool)

        threshold = max(theta - self.offset, self.min_threshold)
        return window_int > threshold
