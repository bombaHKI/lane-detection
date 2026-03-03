# Thesis Project: Lane detection from LIDAR data

This repository is part of a thesis project aimed at creating "ground truth" map annotations.

**Input**: LIDAR data of roads.
**Expected Output**: Lane dividers in wkt format.

### Steps
The planned vague steps for creating lane dividers from LIDAR data:
1. **Segment Ground**: Discard noise, keep points that are part of the road.
2. **Segment Road**: Discard points that are not part of the road.
3. **Segment Road Markings**
4. **Create Bird Eye View**
5. **Detect Lane Borders**

Further details and progress will be documented as the project evolves.

### Instructions

### Lidar ept visualization
The distortion around the area is ~1.5 compared to on equator in the epsg3857 coordinates.
For visualising this large lidar input I use use Potree viewer.
The laz files have to be converted to ept tiles first.

**Creating the ept tiles:** `entwine build -i 871e1d886ffffff_cegl_m4_2_ground.laz -o ../lidar_ept/871e1d886ffffff_cegl_m4_2_ground`

(`export DYLD_LIBRARY_PATH=/usr/local/lib:$DYLD_LIBRARY_PATH` for entwine linking issue)
(since then I run: `sudo install_name_tool -add_rpath /usr/local/lib /usr/local/bin/entwine`)

Reminder on how to view the ept files:
- From `data/lidar_ept`: `lidar_ept % http-server -p 8000 --cors`
- From 'projects/potree/' (external): `npm start`
- Open 'http://localhost:1234/examples/lidar_vis.html'.


**Viewing with cesium map**
In order to visualise the lidar with the cesium map, first we have to use a local projection (`EPSG:23700` for Hungary)
1. convert las to local projection: `pdal translate 871e1d886ffffff_cegl_m4_2.laz cegl_m4_2_eov.laz reprojection --filters.reprojection.in_srs="EPSG:3857" --filters.reprojection.out_srs="EPSG:23700"`
2. `entwine build -i cegl_m4_2_eov.laz -o ../lidar_ept/cegl_m4_2_eov`
3. in the potree html file set: `pointcloudProjection = "+proj=somerc +lat_0=47.1443937222222 +lon_0=19.0485717777778 +k_0=0.99993 +x_0=650000 +y_0=200000 +ellps=GRS67 +towgs84=52.17,-71.82,-14.9,0,0,0,0 +units=m +no_defs";`

**Viewing geopackage**
First we need to create geopackage files from the geojson, then load it potree.
Example in `lidar_vis_markings.html`

		

### Project Management
Using the credentials for uv: `export $(grep -v '^#' .secrets.env | xargs)`