"""Plane filter stage using RANSAC per window.

- For each window, select the 10%-80% quantile of points by z value.
- Fit a plane (z = ax + by + c) using RANSAC.
- Drop all window points further than window_size / 15 from the plane.
"""
from __future__ import annotations

import numpy as np
from lane_detection.processors.base import Processor
from lane_detection.utils.plane_fitting import fit_plane_ransac

class PlaneFilterStage(Processor):
    def __init__(self, quantile_lo: float = 0.1, quantile_hi: float = 0.7):
        super().__init__()
        self.quantile_lo = quantile_lo
        self.quantile_hi = quantile_hi

    def process_window(self, bin_indices: np.ndarray, indices: np.ndarray, context) -> np.ndarray:
        if indices.size < 3:
            return np.ones(indices.size, dtype=bool)
        las = context.las
        xs = np.asarray(las.x, dtype=np.float64)[indices]
        ys = np.asarray(las.y, dtype=np.float64)[indices]
        zs = np.asarray(las.z, dtype=np.float64)[indices]
        # Select quantile
        z_lo = np.quantile(zs, self.quantile_lo)
        z_hi = np.quantile(zs, self.quantile_hi)
        mask = (zs >= z_lo) & (zs <= z_hi)
        if mask.sum() < 3:
            return np.ones(indices.size, dtype=bool)
        pts = np.column_stack([xs[mask], ys[mask], zs[mask]])
        res = fit_plane_ransac(pts, residual_threshold=0.05)
        if res is None:
            self.logger.info("Open3D plane fit failed; keeping all points.")
            return np.ones(indices.size, dtype=bool)
        a, b, c, _ = res
        # Distance to plane
        dists = np.abs(a * xs + b * ys + c - zs) / np.sqrt(a**2 + b**2 + 1)
        threshold = context.window_size * context.bin_length / 25.0
        keep = dists <= threshold
        return keep
