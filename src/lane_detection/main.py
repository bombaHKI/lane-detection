import laspy
from pathlib import Path
from functools import partial

from lane_detection.utils.logger import create_logger
from lane_detection.pipeline.pipeline import Pipeline, Context
from lane_detection.car_trace.trace_stage import CarTraceStage
from lane_detection.binning.binner import BinningStage
from lane_detection.processors.plane_filter import PlaneFilterStage
from lane_detection.processors.trace_distance_filter import TraceDistanceFilterStage
from lane_detection.processors.ground_segment import GroundSegmentStage
from lane_detection.processors.intensity_threshold import IntensityThresholdStage

from lane_detection.io.setup import SetupOutputStage
from lane_detection.io.writer import WriteCloudStage

logger = create_logger('Main')

def load_config(config_file_path):
    """Load the configurations from a config file"""
    return {
        'point_path': 'data/LiDaR/871e1d8b1ffffff_cegl_m4_1.laz',
        # 'point_path': 'data/LiDaR/871e1d886ffffff_cegl_m4_2.laz',
        # 'point_path': 'data/output/LiDaR/distance_clip/distance_clip_in.laz',
        # 'point_path': 'data/output/LiDaR/threshold/intensity_in.laz',
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
        CarTraceStage('window_median_v2', 'data/output/car_trace/trace_window_median_v2.npz'),
        BinningStage(
            time_treshold=config['binning_time_treshold']
        ),
        # TraceDistanceFilterStage(config['distance_filter']),
        # WriteCloudStage('distance'),
        PlaneFilterStage(),
        WriteCloudStage('plane_prefilter'),
        GroundSegmentStage(square_size=2, method='grad', distance_threshold=0.05),
        WriteCloudStage('ground'),
        IntensityThresholdStage(offset=0),
        WriteCloudStage('intensity'),
    ])


def main():
    logger.info("Starting Pipeline")
    config = load_config("config/config.yaml")
    las = read_point_cloud(config["point_path"])
    context = Context(las, config=config)
    context.window_size = config['window_size']
    context.window_shift = config['window_shift']
    context.bin_length = config['bin_length']

    pipeline = build_pipeline(config)

    pipeline.run(context)

if __name__ == '__main__':
    main()