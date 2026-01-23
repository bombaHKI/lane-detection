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
For visualising this large lidar input I use use Potree viewer.
The laz files have to be converted to ept tiles first.

**Creating the ept tiles:** `entwine build -i 871e1d886ffffff_cegl_m4_2_ground.laz -o ../lidar_ept/871e1d886ffffff_cegl_m4_2_ground`

Reminder on how to view the ept files:
- From 'data/lidar_ept': 'lidar_ept % http-server -p 8000 --cors'
- From 'projects/potree/' (external): 'npm start'
- Open 'http://localhost:1234/examples/lidar_vis.html'.

### Project Management
Using the credentials for uv: `export $(grep -v '^#' .secrets.env | xargs)`