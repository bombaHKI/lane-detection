import laspy
import json
import sys
from pathlib import Path

from lane_detection.utils.logger import create_logger
from lane_detection.pipeline.pipeline import Pipeline, Context
from lane_detection.car_trace.trace_stage import CarTraceStage
from lane_detection.binning.binner import BinningStage
from lane_detection.processors.plane_filter import PlaneFilterStage
from lane_detection.processors.trace_distance_filter import TraceDistanceFilterStage
from lane_detection.processors.ground_segment import GroundSegmentStage
from lane_detection.processors.intensity_threshold import IntensityThresholdStage
from lane_detection.processors.road_surface import RoadSurfaceFilter
from lane_detection.processors.local_sor import LocalSORStage
from lane_detection.processors.downsample import DownsampleStage
from lane_detection.processors.fit_lines import FitLinesStage
from lane_detection.processors.dbscan_filter import DBSCANFilterStage
from lane_detection.io.setup import SetupOutputStage
from lane_detection.io.writer import WriteCloudStage, WriteLines

logger = create_logger("Main")

def load_config(config_arg: str | None = None) -> dict:
    """Load pipeline config JSON from config/.

    Selection order:
    1) explicit argument (CLI)
    2) default file: config/default.json

    If a bare name is passed (e.g. "cegl_fast"), it resolves to
    config/cegl_fast.json.
    """
    config_dir = Path("config")
    if config_arg is None:
        path = config_dir / "default.json"
    else:
        candidate = Path(config_arg)
        if candidate.suffix:
            path = candidate
        else:
            path = config_dir / f"{config_arg}.json"

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    logger.info(f"Using config: {path}")
    return config


STAGE_REGISTRY = {
    "setup_output": SetupOutputStage,
    "car_trace": CarTraceStage,
    "binning": BinningStage,
    "trace_distance_filter": TraceDistanceFilterStage,
    "plane_filter": PlaneFilterStage,
    "ground_segment": GroundSegmentStage,
    "intensity_threshold": IntensityThresholdStage,
    "road_surface": RoadSurfaceFilter,
    "local_sor": LocalSORStage,
    "downsample": DownsampleStage,
    "fit_lines": FitLinesStage,
    "dbscan_filter": DBSCANFilterStage,
    "write_cloud": WriteCloudStage,
    "write_lines": WriteLines,
}


def read_point_cloud(lidar_path: str):
    logger.info(f"Reading LAS from: {lidar_path}")
    return laspy.read(lidar_path)


def build_pipeline(config: dict) -> Pipeline:
    stages = []
    for i, stage_cfg in enumerate(config["stages"]):
        if not stage_cfg.get("enabled", True):
            continue

        stage_name = stage_cfg["name"]
        params = stage_cfg.get("params", {})
        stage_cls = STAGE_REGISTRY.get(stage_name)
        if stage_cls is None:
            raise ValueError(f"Unknown stage at index {i}: {stage_name!r}")

        stages.append(stage_cls(**params))

    return Pipeline(stages)


def main():
    config_arg = sys.argv[1] if len(sys.argv) > 1 else None
    config = load_config(config_arg)
    logger.info("Starting pipeline")

    las = read_point_cloud(config["input"]["point_path"])
    context = Context(las, config=config)
    context.window_size = config["context"]["window_size"]
    context.window_shift = config["context"]["window_shift"]
    context.bin_length = config["context"]["bin_length"]

    pipeline = build_pipeline(config)
    pipeline.run(context)


if __name__ == "__main__":
    main()