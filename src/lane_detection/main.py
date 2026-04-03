import laspy
from lane_detection.utils.logger import create_logger

from lane_detection.pipeline.pipeline import Pipeline, Context, Stage, SubStage

logger = create_logger('Main')

def load_config(config_path):
    return {
        'point_path': 'data/LiDaR/871e1d886ffffff_cegl_m4_2.laz'
    }

def read_point_cloud(lidar_path):
    logger.info(f'Reading to las from: {lidar_path}')
    las = laspy.read(lidar_path)
    return las

def build_pipeline(config):
    # trace_algo = create_trace_algorithm(config["trace"])

    return Pipeline([
        Stage(),
        SubStage(),
        Stage(),
        # TraceStage(trace_algo),

        # WriteStage("after_trace"),

        # BinningStage(**config["binning"]),
        # WriteStage("after_binning"),

        # SORProcessor(**config["sor"]),
        # WriteSplitStage("sor", mode="delta"),

        # GroundProcessor(),
        # WriteSplitStage("ground", mode="delta"),

        # WritePlanesStage(),

        # ProjectionStage(**config["projection"]),
        # LaneDetectionStage(),
    ])

def main():
    logger.info("Starting main pipeline")
    config = load_config("config/config.yaml")
    point_cloud = read_point_cloud(config["point_path"])
    context = Context(point_cloud)
    pipeline = build_pipeline(config)
    pipeline.run(context)
    
if __name__ == '__main__':
    main()