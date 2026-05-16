import glob
from pathlib import Path
import numpy as np
import geopandas as gpd
import shapely

npz_files = glob.glob("data/output/05_16_14_12/trace_window_median_v2.npz")
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
    print(f"Saved: {gpkg_path}")
