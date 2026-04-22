import numpy as np

from lane_detection.pipeline.pipeline import Stage

class Processor(Stage):
    def process_window(self, indices: np.ndarray, context) -> np.ndarray:
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
            combined = np.concatenate(bins[start : start + window_size])
            windows.append(np.unique(combined))
            start += window_shift

        n_points = len(context.las.x)
        global_mask = np.zeros(n_points, dtype=bool)

        for window_indices in windows:
            if window_indices.size == 0:
                continue
            keep_mask = self.process_window(window_indices, context)
            global_mask[window_indices[keep_mask]] = True

        context.global_mask = global_mask
        context.bins = [indices[global_mask[indices]] for indices in bins]