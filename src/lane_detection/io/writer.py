import laspy

from lane_detection.pipeline.pipeline import Stage

class WriteCloudStage(Stage):
    def __init__(self, file_name, path = "data/LiDaR/"):
        self.path = path
        self.file_name = file_name

    def run(self, context):
        las = context.las

        in_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        in_points.points = las.points[context.mask]
        in_points.write(self.path + f"{self.file_name}_in.laz")

        out_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
        out_points.points = las.points[~context.mask]
        out_points.write(self.path + f"{self.file_name}_out.laz")