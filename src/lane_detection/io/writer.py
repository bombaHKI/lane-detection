import laspy

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.logger import create_logger

logger = create_logger('Writing Stage')

class WriteCloudStage(Stage):
    def __init__(self, file_name, path = "data/output/LiDaR/"):
        self.path = path
        self.file_name = file_name

    def run(self, context):
        logger.info("Writing to files")
        las = context.las

        in_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        in_points.points = las.points[context.global_mask]
        inlier_name = self.path + f"{self.file_name}_in.laz"
        in_points.write(inlier_name)
        logger.info(f"Written to: {inlier_name}")

        out_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        out_points.points = las.points[~context.global_mask]
        outlier_name = self.path + f"{self.file_name}_out.laz"
        out_points.write(outlier_name)
        logger.info(f"Written to: {outlier_name}")