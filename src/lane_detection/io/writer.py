import laspy

from lane_detection.pipeline.pipeline import Stage, Context
from lane_detection.utils.logger import create_logger

logger = create_logger('Writing Stage')

class WriteCloudStage(Stage):
    def __init__(self, file_name, all=False):
        self.file_name = file_name
        self.all = all

    def run(self, context: Context):
        logger.info("Writing to files")
        output_dir = context.output_dir
        las = context.las

        if context.global_mask is None or self.all:
            logger.info("Writing all points. (global_mask is not set or `all` passed as param)")
            inlier_name = output_dir / f"{self.file_name}_all.laz"
            las.write(inlier_name)
            logger.info(f"Written to: {inlier_name}")
            return

        try:
            in_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
            in_points.points = las.points[context.global_mask]
            inlier_name = output_dir / f"{self.file_name}_in.laz"
            if len(in_points.points) > 1:
                in_points.intensity[0] = 1
                in_points.intensity[1] = 255
            in_points.write(inlier_name)
            logger.info(f"Written to: {inlier_name}")

            out_mask = ~context.global_mask
            if context.prev_mask is not None:
                out_mask &= context.prev_mask
            out_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
            out_points.points = las.points[out_mask]
            if len(out_points.points) > 1:
                out_points.intensity[0] = 0
                out_points.intensity[1] = 255
            outlier_name = output_dir / f"{self.file_name}_out.laz"
            out_points.write(outlier_name)
            logger.info(f"Written to: {outlier_name}")
        except Exception as e:
            logger.info(f"Failed to write point clouds: {e}")