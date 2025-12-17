# Cartesian Planner (Spline)

This package provides a ROS 2 action server that converts an end-effector–frame goal pose into a spline of waypoints and executes them through `ArmControl`.

## Nodes

- `spline_planner` (`cartesian_planner/spline_planner.py`): Action server `spline_plan` (action: `cartesian_planner/PlanSpline`). Goal pose is expressed in the EE frame; the node looks up the current EE pose, transforms the goal to the base frame, builds a cubic spline for translation, keeps orientation fixed, converts to relative deltas, and feeds them sequentially to `ArmControl`.



## Usage
- Run [Eddie-Ros](https://github.com/Robots4Sustainability/eddie-ros/tree/dev)

- Run the planner:
  ```
  ros2 run cartesian_planner spline_planner
  ```

- CLI test (send a +5 cm EE-frame move in Z):
  ```
  ros2 action send_goal spline_plan cartesian_planner/action/PlanSpline "{ target_pose: { position: {x: 0.0, y: 0.0, z: 0.05}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0} } }"
  ```


The action result returns `success`/`message`; feedback publishes progress (0–1). Relative waypoints are sent directly to `right_arm/arm_control`.

## Raster Scan

The `spline_planner` also exposes a service to perform a raster scan motion relative to a center pose.

- **Service**: `/plan_scan_path` (`cartesian_planner/srv/PlanScanPath`)
- **Example**:
  ```bash
  ros2 service call /plan_scan_path cartesian_planner/srv/PlanScanPath "{center_pose: {header: {frame_id: eddie_base_link}, pose: {position: {x: 0.5, y: -0.3, z: 0.7}, orientation: {w: 1.0}}}, width: 0.5, height: 0.2, spacing: 0.05, line_spacing: 0.1}"
  ```

### Parameters

- **center_pose**: The center point `(x, y, z)` of the scan pattern.
  - The scan is generated in the **Y-Z plane** of the base frame at the specified `x` depth.
- **width**: Total extent of the scan area along the **Y-axis** (horizontal).
- **height**: Total extent of the scan area along the **Z-axis** (vertical).
- **spacing**: Distance between waypoints along each horizontal line (scan resolution).
- **line_spacing**: Vertical distance between horizontal scan lines.

