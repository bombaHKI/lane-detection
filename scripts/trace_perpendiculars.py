import geopandas as gpd
import numpy as np
import shapely

gdf = gpd.read_file('../data/output/car_trace/geojson/median_v2.geojson')
gdf = gdf.set_crs(epsg=3857)

trajectory = shapely.LineString([(x, y) for x, y, *_ in gdf.iloc[0].geometry.coords])
steps = 30
eps = 1
points = [[trajectory.interpolate(distance+offset) for offset in (-eps, 0, eps)] for distance in np.arange(steps,trajectory.length,steps)]

perps = []
ends = []
perp_len = 30
for a,b,c in points:
    ac = shapely.LineString([a,c])
    left = ac.parallel_offset(perp_len / 2, 'left')
    right = ac.parallel_offset(perp_len / 2, 'right')
    perp = shapely.LineString([left.interpolate(0.5,normalized=True), right.interpolate(0.5,normalized=True)])
    perps.append(perp)
    ends.append(left)
    ends.append(ac)
    ends.append(right)

perps_gdf = gpd.GeoDataFrame(
    {"id": range(len(perps))},
    geometry=perps,
    crs=gdf.crs
)
perps_gdf.to_file("../data/output/car_trace/geopackage/kecs_perps.gpkg", layer="perps", driver="GPKG")

ends_gdf = gpd.GeoDataFrame(
    {"id": range(len(ends))},
    geometry=ends,
    crs=gdf.crs
)
ends_gdf.to_file("../data/output/car_trace/geopackage/kecs_ends.gpkg", layer="ends", driver="GPKG")

gdf.to_file("../data/output/car_trace/geopackage/kecs_median_v2.gpkg", driver="GPKG")

