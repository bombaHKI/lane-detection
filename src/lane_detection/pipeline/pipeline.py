class Pipeline:
    def __init__(self, stages):
        self.stages = stages

    def run(self, context):
        for stage in self.stages:
            stage.run(context)
        
class Stage:
    def run(self, context):
        raise NotImplementedError
    
class Context:
    def __init__(self, las):
        self.las = las

        # window (number of bins)
        self.window_size = None
        self.window_shift = None

        # geometry
        self.trace = None
        self.bins = None
        self.windows = None

        # masks
        self.global_mask = None
        self.prev_mask = None