import numpy as np

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.logger import create_logger
from lane_detection.pipeline.pipeline import Context

class Processor(Stage):
    def __init__(self):
        super().__init__()

    def process_window(self, bin_indices: np.ndarray, point_indices: np.ndarray, context: Context) -> np.ndarray:
        """Process one window.

        Parameters
        ----------
        bin_indices : np.ndarray of int
            Indices into ``context.bins`` identifying which bins form this window.
        point_indices : np.ndarray of int
            Unique LAS point indices of the combined bins in this window.
        context : Context

        Returns
        -------
        np.ndarray of bool
            Boolean keep-mask over ``point_indices``.
            ``True`` = keep the point.
        """
        raise NotImplementedError

    def run(self, context):
        bins = context.bins
        window_size = context.window_size
        window_shift = context.window_shift

        # Build windows as arrays of bin indices.
        windows: list[np.ndarray] = []
        start = 0
        while start + window_size <= len(bins):
            windows.append(np.arange(start, start + window_size, dtype=np.intp))
            start += window_shift

        n_windows = len(windows)
        self.logger.info(f"Processing {n_windows} windows (size={window_size}, shift={window_shift}).")

        n_points = len(context.las.x)
        global_mask = np.zeros(n_points, dtype=bool)
        log_step = max(1, n_windows // 10)

        for i, bin_indices in enumerate(windows):
            if i % log_step == 0 or i == n_windows - 1:
                self.logger.info(f"Window {i + 1}/{n_windows} ({(i + 1) / n_windows * 100:.0f}%)")
            point_indices = np.unique(np.concatenate([bins[j].indices for j in bin_indices]))
            if point_indices.size == 0:
                continue
            keep_mask = self.process_window(bin_indices, point_indices, context)
            global_mask[point_indices[keep_mask]] = True

        kept = int(global_mask.sum())
        prev_size = n_points
        if context.prev_mask is not None:
            prev_size = int(np.sum(context.prev_mask))
        self.logger.info(f"Processor done: kept {kept}/{prev_size} points.")

        context.prev_mask = context.global_mask
        context.global_mask = global_mask
        context.bins = [b.with_indices(b.indices[global_mask[b.indices]]) for b in bins]