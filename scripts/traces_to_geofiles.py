import glob
from pathlib import Path
import numpy as np
import geopandas as gpd
import shapely

npz_files = glob.glob("data/output/*/*.npz")
print(f"Found {len(npz_files)} trace file(s)")

for npz_path in npz_files:
    npz_path = Path(npz_path)
    data = np.load(npz_path)
    positions = data["positions"]  # (N, 3) array of x, y, z

    line = shapely.LineString(positions)  # use x, y only for 2D geometry
    gdf = gpd.GeoDataFrame(
        {"name": [npz_path.stem]},
        geometry=[line],
        crs="EPSG:3857",
    )

    gpkg_path = npz_path.with_suffix(".gpkg")
    gdf.to_file(gpkg_path, driver="GPKG")
    gjson_path = npz_path.with_suffix(".geojson")
    gdf.to_file(gjson_path, driver="GeoJSON")
    print(f"Saved: {gpkg_path}")
    print(f"Saved: {gjson_path}")
