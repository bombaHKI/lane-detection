from __future__ import annotations

import numpy as np

import shapely
from shapely import contains_xy
from shapely.ops import linemerge

from lane_detection.pipeline.pipeline import Context, Stage
from lane_detection.processors.base import Processor
from lane_detection.utils.grid import build_grid

class RoadSurfaceGround(Stage):
    """Road Surface Ground Stage
    Use after road surface detection and ground detection.

    Method: Reassigns the bins so that they contain all the points that are in the road surface.
    Note: context.ground_bins should be assigned.
    """

    def __init__(self):
        super().__init__()

    def run(self, context: Context):
        self.logger.info("Cropping ground points to Road Surface areas.")
        if context.ground_bins is None:
            raise ValueError("context.ground_bins must not be None")

        bins: list[np.ndarray] = []
        global_mask = np.zeros(len(context.las.x), dtype=bool)
        
        ws = context.window_size
        shift = context.window_shift
        num_windows = len(context.road_surfaces)
        for bin_idx, indices in enumerate(context.ground_bins):
            points = context.las[indices]
            mask = np.zeros(len(indices), dtype=bool)

            first_w = max(0, (bin_idx - ws + 1) // shift)
            last_w = min(num_windows - 1, bin_idx // shift)

            for w in range(first_w, last_w + 1):
                start = w * shift
                end = start + ws

                if start <= bin_idx < end:
                    rs = context.road_surfaces[w]
                    mask |= contains_xy(rs, points.x, points.y)

            keep_indices = indices[mask]
            bins.append(keep_indices)
            global_mask[keep_indices] = True

        context.bins = bins
        context.prev_mask = context.ground_mask
        context.global_mask = global_mask
