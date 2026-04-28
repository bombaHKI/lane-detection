from lane_detection.utils.logger import create_logger

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
        self.bins = None

        # masks
        self.global_mask = None
        self.prev_mask = None