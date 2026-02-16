"""Extract point clouds for specific GPS time values."""

import numpy as np
import laspy
from pathlib import Path


def extract_time_slices(
    input_file: str,
    output_dir: str = "data/LiDaR/time_slices",
    start_index: int = 1000,
    num_slices: int = 10
):
    """
    Extract point clouds for specific unique GPS time values and save as separate LAS files.
    
    Args:
        input_file: Path to input LiDAR file (.laz or .las)
        output_dir: Directory to save output files
        start_index: Index of first unique time value to extract (default: 1000)
        num_slices: Number of time slices to extract (default: 10)
    """
    print(f"Loading point cloud from {input_file}...")
    las = laspy.read(input_file)
    
    if 'gps_time' not in las.point_format.dimension_names:
        raise ValueError(f"No 'gps_time' attribute found in {input_file}")
    
    timestamps = las.gps_time
    print(f"Total points: {len(timestamps)}")
    
    # Get unique time values
    unique_times = np.unique(timestamps)
    print(f"Unique time values: {len(unique_times)}")
    print(f"Time range: {unique_times.min():.3f} to {unique_times.max():.3f}")
    
    # Check if we have enough unique times
    if start_index + num_slices > len(unique_times):
        raise ValueError(
            f"Not enough unique times. Requested indices {start_index}-{start_index+num_slices-1}, "
            f"but only {len(unique_times)} unique values available."
        )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Extract and save slices
    for i in range(num_slices):
        time_index = start_index + i
        target_time = unique_times[time_index]
        
        # Filter points with this exact timestamp
        mask = timestamps == target_time
        num_points = np.sum(mask)
        
        print(f"\nSlice {i}: Time index {time_index}, Time value {target_time:.6f}, Points: {num_points}")
        
        # Create new LAS file with filtered points
        header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
        header.offsets = las.header.offsets
        header.scales = las.header.scales
        
        las_slice = laspy.LasData(header)
        
        # Copy filtered points
        las_slice.x = las.x[mask]
        las_slice.y = las.y[mask]
        las_slice.z = las.z[mask]
        
        # Copy all other attributes
        for dimension in las.point_format.dimension_names:
            if dimension not in ['X', 'Y', 'Z']:
                setattr(las_slice, dimension.lower(), getattr(las, dimension.lower())[mask])
        
        # Save to file
        output_file = output_path / f"time_slice_{time_index:04d}_t{target_time:.3f}.laz"
        las_slice.write(str(output_file))
        print(f"Saved to {output_file}")
    
    print(f"\n✓ Successfully extracted {num_slices} time slices to {output_dir}")


def main():
    """Main entry point."""
    extract_time_slices(
        input_file="data/LiDaR/871e1d886ffffff_cegl_m4_2.laz",
        output_dir="data/LiDaR/time_slices",
        start_index=100000,
        num_slices=10
    )


if __name__ == "__main__":
    main()
