# Lane Detection from LiDAR Data

![Lane geometries drawn on the lidar point cloud](media/lanes_drawn_on_pointcloud.png)

Thesis project for extracting lane-divider geometries from SLAM-stitched, georeferenced LiDAR point clouds.

- **Input**: A large stitched LiDAR scan (`.las/.laz` 100+ Million points)
- **Output**: Lane divider geometries in a GeoPackage, labelled as `solid` or `dashed`

## Pipeline

The pipeline processes the point cloud through the following sequential steps:

1. **Detect Car Trace** — Reconstruct the vehicle trajectory to approximate lane positions. ![Trace clip](./media/0_trace.png)
2. **Distance Clip** - Points that are far from the car trace are discarded. ![Distance clip](./media/1_distance.gif)
3. **Segment Ground** — Filter out noise; retain only ground-level points. ![Ground segment](./media/2_ground.gif)
4. **Intensity Thresholding** — Discard low-intensity points unlikely to be lane markings. ![Intensity thresholding](./media/3_intensity.gif)
5. **Segment Road Markings** — Isolate points on the road surface. ![Road markings](./media/4_road_surface.gif)
6. **Fit and Label lines** — Fit lines to road marking clusters using a line-growing algorithm, then label them. ![Line fit and label](./media/5_lines.gif)



## Running the pipeline

```bash
uv run src/lane_detection/main.py config/full_pipeline.json
```

This will run the pipeline steps on the lidar data defined in the congig file and create an output folder in `data/ouput`.

The pipeline is driven by a JSON config file. Each step is identified by a keyword defined in `main.py` and executed in order, with each step updating a shared pipeline context. See [config/full_pipeline.json](config/full_pipeline.json) for a reference configuration.

> **Note on coordinate systems**: All distance parameters in the config are interpreted in the coordinate system of the input LiDAR file. For data in `EPSG:3857` (Web Mercator), there is a scale distortion relative to real-world distances — approximately 1.5× at Hungarian latitudes. For example, a 15 m clip distance in the config corresponds to roughly 10 m on the ground.

## Visualisation

Point clouds are visualised using [Potree](https://github.com/potree/potree). The `.laz` files must first be converted to EPT tiles. See [potree_vis/README.md](potree_vis/README.md) for instructions.