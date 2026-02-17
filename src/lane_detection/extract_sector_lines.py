"""Extract point clouds for specific GPS time values."""

import numpy as np
import laspy
from pathlib import Path
from collections import defaultdict
import json
from timeit import default_timer as timer

from lane_detection.utils import build_pulse_map

def extract_sector_lines(
    las: laspy.LasData,
    clip_index: int = 0,
    steps: int = 4,
    large_steps: int = 20,
    lines_per_large_step = 10,
    num_steps: int = 10
):
    """
    Extract direction lines of lidar pulses.
    `num_steps` lines are extracted with distance `steps` at every `large_steps` time.
    
    Args:
        las: LasData
        clip_index: Number of pulses to discard from beginning and end (default: 0)
        steps: Pulses between steps are disregarded
        large_steps: Pulses between large steps are disregarded
        steps: Extract only every nth line (default: 4)
    """
    
    timestamps = las.gps_time
    print(f"Total points: {len(timestamps)}")
    
    start = timer()
    print(f"Mapping points to time values")
    time_to_points, unique_times = build_pulse_map(las.xyz, timestamps)
    print(f"Mapping done in {timer()-start} seconds")

    if clip_index > 0:
        unique_times = unique_times[clip_index:-clip_index]
    print(f"Unique time values in clip range: {len(unique_times)}")
    print(f"Time range: {unique_times.min():.3f} to {unique_times.max():.3f}")
    
    if len(unique_times) == 0:
        raise ValueError(f"Not enough unique times.")
    
    lines = []
    # Extract and save slices
    progress = 0
    for time_index in range(0, len(unique_times), steps):
        if time_index / len(unique_times) * 100 >= progress + 5:
            progress += 5
            print(f"Progress: {progress}%")

        target_time = unique_times[time_index]

        # Filter points with this exact timestamp
        points = np.asarray(time_to_points[target_time])
        if points.ndim != 2 or points.shape[1] < 3:
            continue
        num_points = points.shape[0]

        if num_points < 120:
            # print(f"Number of points less than 100, SKIPPING")
            continue
        
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # Linear regression y = m*x + b
        m, b = np.polyfit(x, y, 1)
        z_avg = float(np.mean(z))
        x_min = float(np.min(x)) - 30
        x_max = float(np.max(x)) + 30
        y_min = float(m * x_min + b)
        y_max = float(m * x_max + b)
        
        lines.append({
            "index": len(lines),
            "time": float(target_time),
            "m": float(m),
            "b": float(b),
            "z_avg": z_avg,
            "p1": (x_min, y_min, z_avg),
            "p2": (x_max, y_max, z_avg)
        })

    print(f"Finished in {timer()-start} seconds.")
    print(f"Created {len(lines)} lines")
    return lines


def main():
    """Main entry point."""
    las = laspy.read("data/LiDaR/871e1d886ffffff_cegl_m4_2.laz")


    lines = extract_sector_lines(las, clip_index=1000, steps=100)
    output_dir = "data/car_trace"
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Write each line to its own file (two points only)
    for line in lines:
        line_path = output_path / f"line_{line['index']:06}.geojson"
        x1, y1, z1 = line["p1"]
        x2, y2, z2 = line["p2"]

        geojson = {
            "type": "FeatureCollection",
            "crs": {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:EPSG::3857"
                }
            },
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "index": line["index"],
                        "time": line["time"],
                        "m": line["m"],
                        "b": line["b"],
                        "z_avg": line["z_avg"]
                    },
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [
                            [x1, y1, z1],
                            [x2, y2, z2]
                        ]
                    }
                }
            ]
        }

        with open(line_path, "w", encoding="utf-8") as f:
            json.dump(geojson, f, indent=2)


if __name__ == "__main__":
    main()
