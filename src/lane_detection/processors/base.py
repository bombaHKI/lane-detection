import numpy as np

from lane_detection.pipeline.pipeline import Stage

class Processor(Stage):
    def process_bin(self, points, indices, context):
        raise NotImplementedError

    def run(self, context):
        global_mask = np.zeros(len(context.original_cloud), dtype=bool)

        for bin_indices in context.bins:
            result_mask = self.process_bin(
                context.current_cloud[bin_indices],
                bin_indices,
                context
            )

            global_mask[bin_indices] |= result_mask

        context.global_mask = global_mask
        context.current_cloud = context.original_cloud[global_mask]