import laspy
import geopandas as gpd

file_path_trace = "data/geojson/car_trajectory_v1_window_median_ground_detect_dp_simple_1.geojson"
file_path_laz = "data/LiDaR/871e1d886ffffff_cegl_m4_2.laz"
las = laspy.read(file_path_laz)
gdf = gpd.read_file(file_path_trace)


points = gpd.GeoDataFrame(geometry=gpd.points_from_xy(
    las.x, las.y), crs="EPSG:3857")
print("files read, geodataframe ready")

distances = points.distance(gdf.iloc[0]["geometry"])
print("distances calculated, writing to files")

close_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
close_points.points = las.points[(distances < 21).to_numpy()]
close_points.write("data/LiDaR/871e1d886ffffff_cegl_m4_2_close_to_trace.laz")

far_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
far_points.points = las.points[(distances >= 21).to_numpy()]
far_points.write("data/LiDaR/871e1d886ffffff_cegl_m4_2_far_from_trace.laz")