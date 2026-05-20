# Potree Point Cloud Visualisation

The pointclouds (even multiple ones at the same time 500+ Million points) can be viewed and 
navigated in the browser via [Potree](https://github.com/potree/potree/).

## Building EPT tiles

For this step [entwine is required](https://entwine.io/en/latest/).

From the `data/LiDaR/` directory:

```bash
entwine build -i 871e1d886ffffff.laz -o ../lidar_ept/871e1d886ffffff
```

## Viewing the point cloud

1. Clone the [Potree](https://github.com/potree/potree) repository to a separate location and copy `lidar_vis.html` into its `examples/` folder.
2. Serve the EPT data from `data/`:
   ```bash
   http-server -p 8000 --cors
   ```
3. Start the Potree dev server from the Potree project root:
   ```bash
   npm start
   ```
4. Open `http://localhost:1234/examples/lidar_vis.html` in a browser.

## Adding GeoJson & Geopackage files

See the example in the [example html](./lidar_vis.html).

## Viewing with a Cesium map

Cesium requires a local projection. An example with the Hungarian `EPSG:23700`.

1. Reproject the point cloud:
   ```bash
   pdal translate 871e1d886ffffff_cegl_m4_2.laz cegl_m4_2_eov.laz reprojection \
     --filters.reprojection.in_srs="EPSG:3857" \
     --filters.reprojection.out_srs="EPSG:23700"
   ```
2. Build EPT tiles from the reprojected file:
   ```bash
   entwine build -i cegl_m4_2_eov.laz -o ../lidar_ept/cegl_m4_2_eov
   ```
3. Set the projection string in `lidar_vis.html` (expample in `potree/examples/cesium_retz.html`):
   ```js
   pointcloudProjection = "+proj=somerc +lat_0=47.1443937222222 +lon_0=19.0485717777778 +k_0=0.99993 +x_0=650000 +y_0=200000 +ellps=GRS67 +towgs84=52.17,-71.82,-14.9,0,0,0,0 +units=m +no_defs";
   ```
