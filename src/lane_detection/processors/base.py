import numpy as np

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.logger import create_logger
from lane_detection.pipeline.pipeline import Context

class Processor(Stage):
    def __init__(self):
        super().__init__()

    def process_window(self, indices: np.ndarray, context: Context) -> np.ndarray:
        """Process one window.  Return a boolean mask over *indices* of points
        to keep (True = keep)."""
        raise NotImplementedError

    def run(self, context):
        bins = context.bins
        window_size = context.window_size
        window_shift = context.window_shift

        windows: list[np.ndarray] = []
        start = 0
        while start + window_size <= len(bins):
            combined = np.concatenate([b.indices for b in bins[start : start + window_size]])
            windows.append(np.unique(combined))
            start += window_shift

        n_windows = len(windows)
        self.logger.info(f"Processing {n_windows} windows (size={window_size}, shift={window_shift}).")

        n_points = len(context.las.x)
        global_mask = np.zeros(n_points, dtype=bool)
        log_step = max(1, n_windows // 10)

        for i, window_indices in enumerate(windows):
            if i % log_step == 0 or i == n_windows - 1:
                self.logger.info(f"Window {i + 1}/{n_windows} ({(i + 1) / n_windows * 100:.0f}%)")
            if window_indices.size == 0:
                continue
            keep_mask = self.process_window(window_indices, context)
            global_mask[window_indices[keep_mask]] = True

        kept = int(global_mask.sum())
        prev_size = n_points
        if context.prev_mask is not None:
            prev_size = int(np.sum(context.prev_mask))
        self.logger.info(f"Processor done: kept {kept}/{prev_size} points.")

        context.prev_mask = context.global_mask
        context.global_mask = global_mask
        context.bins = [b.with_indices(b.indices[global_mask[b.indices]]) for b in bins]