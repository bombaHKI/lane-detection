from __future__ import annotations

import numpy as np
import laspy

from lane_detection.pipeline.pipeline import Stage, Context
from lane_detection.utils.grid import build_grid


class DownsampleStage(Stage):
    """
    Downsamples the current point cloud using a square grid. Each cell produces one
    averaged point (mean of x, y, z, and intensity). Creates a brand-new LAS object
    and resets all derived context state (global_mask, prev_mask, ground_mask, bins, etc.).
    """

    def __init__(self, square_size: float):
        super().__init__()
        self.square_size = square_size

    def run(self, context: Context):
        las = context.las

        if context.global_mask is not None:
            active_idx = np.nonzero(context.global_mask)[0]
        else:
            active_idx = np.arange(len(las.x), dtype=np.int64)

        n_active = len(active_idx)
        if n_active == 0:
            self.logger.info("No active points; skipping downsample.")
            return

        self.logger.info(f"Downsampling {n_active} points with square_size={self.square_size}.")

        xyz = np.column_stack([
            np.asarray(las.x, dtype=np.float64)[active_idx],
            np.asarray(las.y, dtype=np.float64)[active_idx],
            np.asarray(las.z, dtype=np.float64)[active_idx],
        ])
        intensity = np.asarray(las.intensity, dtype=np.float64)[active_idx]
        gps_time = np.asarray(las.gps_time, dtype=np.float64)[active_idx]

        grid = build_grid(xyz, self.square_size)
        n_cells = len(grid)
        self.logger.info(f"Grid has {n_cells} cells.")

        avg_x = np.empty(n_cells, dtype=np.float64)
        avg_y = np.empty(n_cells, dtype=np.float64)
        avg_z = np.empty(n_cells, dtype=np.float64)
        avg_intensity = np.empty(n_cells, dtype=np.float64)
        avg_gps_time = np.empty(n_cells, dtype=np.float64)

        for i, indices in enumerate(grid.values()):
            avg_x[i] = xyz[indices, 0].mean()
            avg_y[i] = xyz[indices, 1].mean()
            avg_z[i] = xyz[indices, 2].mean()
            avg_intensity[i] = intensity[indices].mean()
            avg_gps_time[i] = gps_time[indices].mean()

        new_las = laspy.create(
            point_format=las.header.point_format,
            file_version=las.header.version,
        )
        new_las.x = avg_x
        new_las.y = avg_y
        new_las.z = avg_z
        new_las.intensity = np.round(avg_intensity).astype(np.uint16)
        new_las.gps_time = avg_gps_time

        self.logger.info(f"Downsampled from {n_active} to {n_cells} points.")

        context.las = new_las
        context.global_mask = np.ones(n_cells, dtype=bool)
        context.prev_mask = None
        context.ground_mask = None
        context.bins = None
        context.ground_bins = None
