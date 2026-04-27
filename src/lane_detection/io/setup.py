import json
from datetime import datetime
from pathlib import Path

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.logger import create_logger

logger = create_logger('Setup Stage')


class SetupOutputStage(Stage):
    def __init__(self, base_path="data/output"):
        self.base_path = Path(base_path)

    def run(self, context):
        timestamp = datetime.now().strftime("%m_%d_%H_%M")
        output_dir = self.base_path / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        context.output_dir = output_dir

        if context.config is not None:
            config_path = output_dir / "config.json"
            config_path.write_text(json.dumps(context.config, indent=2))
            logger.info(f"Config saved to: {config_path}")

        logger.info(f"Output directory: {output_dir}")
