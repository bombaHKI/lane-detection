import laspy
from lane_detection.utils.logger import create_logger

from lane_detection.pipeline.pipeline import Pipeline, Context
from lane_detection.car_trace.trace_stage import CarTraceStage, window_median

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
    return Pipeline([
        CarTraceStage(window_median)
    ])


from utils_old import save_trajectory_geojson

def main():
    logger.info("Starting Pipeline")
    config = load_config("config/config.yaml")
    las = read_point_cloud(config["point_path"])
    context = Context(las)

    pipeline = build_pipeline(config)

    pipeline.run(context)

    trace = context.trace
    save_trajectory_geojson(trace,'data/car_trace/geojson/refactor_median.geojson','')


if __name__ == '__main__':
    main()