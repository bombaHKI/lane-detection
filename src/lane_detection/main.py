import laspy
from pathlib import Path
from functools import partial

from lane_detection.utils.logger import create_logger
from lane_detection.pipeline.pipeline import Pipeline, Context
from lane_detection.car_trace.trace_stage import CarTraceStage, window_median, window_median_v2, pulse_lines, closest_point
from lane_detection.binning.binner import BinningStage
from lane_detection.processors.trace_distance_filter import TraceDistanceFilterStage

from lane_detection.io.setup import SetupOutputStage
from lane_detection.io.writer import WriteCloudStage

logger = create_logger('Main')

def load_config(config_file_path):
    """Load the configurations from a config file"""
    return {
        'point_path': 'data/LiDaR/871e1d886ffffff_cegl_m4_2.laz',
        'bin_length': 20,
        'binning_time_treshold': 10,
        'window_size': 2,
        'window_shift': 1,
        'distance_filter': 15,
    }

def read_point_cloud(lidar_path):
    logger.info(f'Reading to las from: {lidar_path}')
    las = laspy.read(lidar_path)
    return las

def build_pipeline(config):
    return Pipeline([
        SetupOutputStage(),
        CarTraceStage(window_median_v2),
        BinningStage(
            bin_length=config['bin_length'],
            time_treshold=config['binning_time_treshold']
        ),
        TraceDistanceFilterStage(config['distance_filter']),
        WriteCloudStage('distance_clip'),
    ])


from lane_detection.utils_old import save_trajectory_geojson

def main():
    logger.info("Starting Pipeline")
    config = load_config("config/config.yaml")
    las = read_point_cloud(config["point_path"])
    context = Context(las, config=config)
    context.window_size = config['window_size']
    context.window_shift = config['window_shift']

    pipeline = build_pipeline(config)

    pipeline.run(context)

    # trace = context.trace
    # save_trajectory_geojson(trace,'data/output/car_trace/geojson/median_v2.geojson','')


if __name__ == '__main__':
    main()