from lane_detection.utils.logger import create_logger

import shapely

class Pipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, context):
        for stage in self.stages:
            stage.run(context)
        
class Stage:
    def __init__(self):
        self.logger = create_logger(self.__class__.__name__)
    def run(self, context):
        raise NotImplementedError

class Bin:
    """A single spatial bin along the car trace.

    Attributes:
        indices:  int64 array of LAS point indices belonging to this bin.
        trace:    shapely.LineString — the trace segment between the two boundary virtual points.
        perp_S:   shapely.LineString — perpendicular line at the start boundary.
        perp_E:   shapely.LineString — perpendicular line at the end boundary.
    """

    def __init__(self, indices, trace_S, trace_E, perp_S, perp_E):
        self.indices = indices
        self.trace_S = trace_S
        self.trace_E = trace_E
        self.perp_S = perp_S
        self.perp_E = perp_E

    def with_indices(self, indices):
        """Return a new Bin with updated indices but the same geometry."""
        return Bin(indices, self.trace, self.perp_S, self.perp_E)
class Context:
    def __init__(self, las, config=None):
        self.las = las
        self.config = config

        # output
        self.output_dir = None

        # window (number of bins)
        self.window_size = None
        self.window_shift = None
        self.bin_length = None

        # geometry
        self.trace = None
        self.bins: list[Bin] = []
        self.ground_bins: list[Bin] = []
        self.road_surfaces: shapely.Polygon = []

        # masks
        self.global_mask = None
        self.prev_mask = None
        self.ground_mask = None

        # fitted lane lines (list of shapely.LineString)
        self.lines = []