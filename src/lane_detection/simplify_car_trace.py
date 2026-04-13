from pathlib import Path

import geopandas as gpd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

gdf = gpd.read_file(DATA_DIR / "car_trace/geojson/car_trajectory_v1_window_median_ground_detect.geojson")
gdf["geometry"] = gdf.simplify(tolerance=1)
gdf.to_file(DATA_DIR / "car_trace/geojson/car_trajectory_v1_window_median_ground_detect_dp_simple_1.geojson", driver='GeoJSON')
gdf.to_file(DATA_DIR / "car_trace/geopackage/car_trajectory_v1_window_median_ground_detect_dp_simple_1.gpkg", driver='GPKG')