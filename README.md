# Cartesian Planner

This package provides a standalone ROS 2 service that converts absolute end-effector
targets into a sequence of waypoints using spherical linear interpolation (SLERP).

## Node Overview

`slerp_planner` is a Python node (`cartesian_planner/slerp_planner.py`) that exposes a
single ROS 2 service:

```
/plan_cartesian_path (cartesian_planner/srv/PlanCartesianPath)
```

The request contains:

- `target_pose` – absolute goal pose, expressed in any TF frame (default
  `eddie_base_link`).
- `max_translation_step` – maximum linear distance for each waypoint.
- `max_rotation_step` – maximum angular change per waypoint (radians).

The response returns:

- `relative_waypoints` – a list of incremental poses that can be sent directly to the
  Eddie arm `ArmControl` action (which expects relative offsets).
- `success`/`message` – status of the planning call.


## Usage

### Standalone

```
ros2 run cartesian_planner slerp_planner
```

**!! Please try only in simulation !!** :

```
ros2 launch eddie_ros eddie.launch.py use_sim:=true arm_select:=right
```
```
ros2 launch eddie_ros rviz.launch.py
```

**please ensure branch : `dsl/pick_place_fsm` for Finite-State-Machine**

```
ros2 run pick_place_fsm pick_place_fsm_mock
```
publish Pose

```
ros2 topic pub --once /perception/target_pose geometry_msgs/msg/PoseStamped "{header: {frame_id: eddie_base_link }, pose: {position: {x: 0, y: 0.5, z: 0}, orientation: {x: 0.0, y: 0.0, z: 0.0, w: 1.0}}}"

```

The service returns the incremental waypoints; a client can feed these to the
`right_arm/arm_control` action in sequence.


