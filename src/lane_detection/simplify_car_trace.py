import geopandas as gpd

gdf = gpd.read_file("../../data/geojson/car_trajectory_v1_window_median_ground_detect.geojson")
gdf["geometry"] = gdf.simplify(tolerance=1)
gdf.to_file("../../data/geojson/car_trajectory_v1_window_median_ground_detect_dp_simple_1.geojson", driver='GeoJSON')
gdf.to_file("../../data/geopackage/car_trajectory_v1_window_median_ground_detect_dp_simple_1.gpkg", driver='GPKG')