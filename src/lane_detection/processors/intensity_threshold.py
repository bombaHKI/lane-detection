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

logger = create_logger('Intensity Threshold Stage')


def _kapur_threshold(intensities: np.ndarray) -> float | None:
    """Return the max-entropy intensity threshold (Kapur's method) or None."""
    if intensities.size == 0:
        return None
    val, counts = np.unique(intensities, return_counts=True)
    if val.size < 2:
        return None

    probs = counts.astype(np.float64) / counts.sum()
    P_cum = np.cumsum(probs)
    log_probs = np.log(probs, out=np.zeros_like(probs), where=probs != 0)
    H_cum = np.cumsum(probs * -log_probs)
    H_total = H_cum[-1]

    eps = 1e-10
    P_omega     = P_cum
    P_omega_bar = 1.0 - P_cum

    term1 = np.log(P_omega + eps)     + (H_cum / (P_omega + eps))
    term2 = np.log(P_omega_bar + eps) + ((H_total - H_cum) / (P_omega_bar + eps))
    Phi = term1 + term2
    Phi[0]  = -np.inf
    Phi[-1] = -np.inf

    return float(val[int(np.argmax(Phi))])


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
        self.offset = float(offset)
        self.min_threshold = float(min_threshold)

    def process_window(self, indices: np.ndarray, context) -> np.ndarray:
        intensities = np.asarray(context.las.intensity)
        window_int = intensities[indices]

        theta = _kapur_threshold(window_int)
        if theta is None:
            return np.ones(indices.size, dtype=bool)

        threshold = max(theta - self.offset, self.min_threshold)
        return window_int > threshold
