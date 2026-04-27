"""Per-cell intensity thresholding using Kapur's maximum entropy method.

Each active point is binned into a 2-D grid (default 1x1 m). Within each cell,
Kapur's max-entropy threshold is computed from the intensity histogram and
points with intensity above ``theta_max - offset`` (clamped to a small minimum)
are kept.
"""
from __future__ import annotations

import numpy as np

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.grid import build_grid
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


class IntensityThresholdStage(Stage):
    """Keep points whose intensity exceeds the per-cell Kapur threshold.

    Parameters
    ----------
    square_size : float, default 1.0
        Edge length of each grid cell (metres).
    offset : float, default 20.0
        Subtracted from each cell's threshold (so more points are kept).
    min_threshold : float, default 20.0
        Lower clamp for the per-cell threshold after applying ``offset``.
    """

    def __init__(
        self,
        square_size: float = 1.0,
        offset: float = 20.0,
        min_threshold: float = 20.0,
    ):
        self.square_size = float(square_size)
        self.offset = float(offset)
        self.min_threshold = float(min_threshold)

    def run(self, context):
        logger.info(
            f"Intensity threshold: square_size={self.square_size}, "
            f"offset={self.offset}, min_threshold={self.min_threshold}"
        )
        las = context.las
        xyz = las.xyz
        intensities = np.asarray(las.intensity)

        if context.global_mask is not None:
            active_idx = np.nonzero(context.global_mask)[0]
        else:
            active_idx = np.arange(len(xyz), dtype=np.int64)

        if active_idx.size == 0:
            logger.info("No active points; skipping.")
            return

        active_pts = xyz[active_idx]
        active_int = intensities[active_idx]

        grid = build_grid(active_pts, self.square_size)
        logger.info(f"Grid built: {len(grid)} cells over {active_idx.size} points.")

        keep_local = np.zeros(active_idx.size, dtype=bool)
        n = len(grid)
        log_step = max(1, n // 10)

        for i, (cell, indices) in enumerate(grid.items()):
            if i % log_step == 0 or i == n - 1:
                logger.info(f"Thresholding cells: {i + 1}/{n} ({(i + 1) / n * 100:.0f}%)")
            cell_int = active_int[indices]
            theta = _kapur_threshold(cell_int)
            if theta is None:
                continue
            threshold = max(theta - self.offset, self.min_threshold)
            keep_local[indices[cell_int > threshold]] = True

        global_mask = np.zeros(len(xyz), dtype=bool)
        global_mask[active_idx[keep_local]] = True

        kept = int(global_mask.sum())
        logger.info(f"Intensity threshold kept {kept}/{active_idx.size} points.")

        context.prev_mask = context.global_mask
        context.global_mask = global_mask
        if context.bins is not None:
            context.bins = [idx[global_mask[idx]] for idx in context.bins]
