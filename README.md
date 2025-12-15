# Cartesian Planner (Spline)

This package provides a ROS 2 action server that converts an end-effector–frame goal pose into a spline of waypoints and executes them through `ArmControl`.

## Nodes

- `spline_planner` (`cartesian_planner/spline_planner.py`): Action server `spline_plan` (action: `cartesian_planner/PlanSpline`). Goal pose is expressed in the EE frame; the node looks up the current EE pose, transforms the goal to the base frame, builds a cubic spline for translation, keeps orientation fixed, converts to relative deltas, and feeds them sequentially to `ArmControl`.



## Usage

- Run the planner:
  ```
  ros2 run cartesian_planner spline_planner
  ```

- CLI test (send a +5 cm EE-frame move in Z):
  ```
  ros2 action send_goal spline_plan cartesian_planner/action/PlanSpline "{ target_pose: { position: {x: 0.0, y: 0.0, z: 0.05}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0} } }"
  ```


The action result returns `success`/`message`; feedback publishes progress (0–1). Relative waypoints are sent directly to `right_arm/arm_control`.
