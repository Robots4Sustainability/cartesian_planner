# Cartesian Planner (Raster Scanner)

This package provides the raster-scan node used by the door-disassembly workflow.

## Main node

- `raster_scanner` (`src/raster_scanner.py`)

Run it with:

```bash
ros2 run cartesian_planner raster_scanner
```

## What it does

The node exposes a raster-scan service and executes the scan through `ArmControl`.

At a high level it:

1. receives four corner poses
2. generates a raster path between those corners
3. sends waypoints to `right_arm/arm_control`
4. after each waypoint, requests perception with:
   - `task_name = detect_screws`
5. stores any detected screw poses
6. returns the collected screw poses in the scan service response

If perception finds no screw at a waypoint, that is treated as a normal result, not as a scan failure.

## Raster generation algorithm

The raster path is built from the four corner poses as follows:

1. transform all four corners into the base frame if needed
2. treat `top_left -> bottom_left` as the left scan edge
3. treat `top_right -> bottom_right` as the right scan edge
4. interpolate points along both vertical edges using SciPy linear interpolation (`scipy.interpolate.interp1d`) and the configured line spacing
5. connect each left/right pair in alternating order to create a zig-zag raster
6. sample that polyline again with SciPy linear interpolation using the waypoint spacing
7. keep the end-effector orientation fixed to the orientation of the starting pose
8. convert each absolute waypoint into a relative move before sending it to `ArmControl`


## Interfaces

### Service

- `/plan_scan_path`
- type: `cartesian_planner/srv/PlanScanPath`

The request uses four corner poses:

- `top_left`
- `top_right`
- `bottom_right`
- `bottom_left`

The response returns:

- `success`
- `message`

When successful, `message` contains JSON with:

```json
{
  "status": "Raster executed",
  "screw_poses": [...]
}
```

### Action used internally

The node uses the perception action server:

- `/run_perception_pipeline`
- type: `my_robot_interfaces/action/RunVision`

It sends:

```text
task_name = detect_screws
object_class = ""
time_duration = 0.0
```

## How door_disassemble uses it

When raster scan is enabled in `door_disassemble`:

1. perception first provides 4 subdoor corner poses
2. `door_disassemble` sends those 4 poses to `/plan_scan_path`
3. `raster_scanner` executes the raster scan
4. screw poses found during the scan are returned in the service response
5. `door_disassemble` stores those poses and later passes them to `screwdriver_pick`

## Notes

- screw detections are de-duplicated by Euclidean distance
- duplicate detections from nearby waypoints are skipped
