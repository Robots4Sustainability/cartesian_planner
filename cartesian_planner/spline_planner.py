#!/usr/bin/env python3
"""
Spline planner with action server:
- Accepts a goal pose expressed in the end-effector frame (relative move).
- Looks up current EE pose, transforms goal to base frame, builds a cubic spline,
  converts absolute samples to relative deltas, and sends them to ArmControl sequentially.
- Can perform a raster scan like motion using Splinewhen triggered with ros service call.
    Configurable paramerters are : pose in base_frame , height , width , space along lines.
    Also publishes a path marker for visualization.
"""

from typing import List

import numpy as np
import rclpy
from rclpy.action import ActionClient, ActionServer
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Pose, PoseStamped
from tf2_ros import Buffer, TransformListener
from cartesian_planner.action import PlanSpline
from cartesian_planner.srv import PlanScanPath
from eddie_ros.action import ArmControl
from scipy.interpolate import CubicSpline
import tf_transformations
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class SplinePlanner(Node):
    def __init__(self) -> None:
        super().__init__("spline_planner")
        self.declare_parameter("base_frame", "eddie_base_link")
        self.declare_parameter("ee_frame", "eddie_right_arm_end_effector_link")
        self.declare_parameter("arm_action_server", "right_arm/arm_control")
        self.declare_parameter("max_translation_step", 0.05)

        self.cb_group = ReentrantCallbackGroup()
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.arm_action_server = self.get_parameter("arm_action_server").value
        self.max_translation_step = float(self.get_parameter("max_translation_step").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.arm_client = ActionClient(self, ArmControl, self.arm_action_server, callback_group=self.cb_group)
        self.path_pub = self.create_publisher(Marker, "/spline_scan_path", 10)

        self.action_server = ActionServer(
            self,
            PlanSpline,
            "spline_plan",
            execute_callback=self.execute_callback,
            callback_group=self.cb_group,
        )
        self.scan_service = self.create_service(PlanScanPath, "plan_scan_path", self.handle_scan_request, callback_group=self.cb_group)

        self.get_logger().info(
            f"Spline planner ready. base_frame={self.base_frame}, ee_frame={self.ee_frame}, arm_server={self.arm_action_server}"
        )

    async def execute_callback(self, goal_handle):
        goal: PlanSpline.Goal = goal_handle.request
        self.get_logger().info("Received spline goal (EE frame).")

        start_pose = self._get_current_pose()
        if start_pose is None:
            goal_handle.abort()
            return PlanSpline.Result(success=False, message="Cannot fetch current pose")

        try:
            goal_st = PoseStamped()
            goal_st.header.frame_id = self.ee_frame
            goal_st.header.stamp = rclpy.time.Time().to_msg()
            goal_st.pose = goal.target_pose
            goal_in_base: PoseStamped = await self._transform_pose(goal_st, self.base_frame)
            goal_base_pose = goal_in_base.pose
        except Exception as exc:
            self.get_logger().error(f"Failed to transform goal to base: {exc}")
            goal_handle.abort()
            return PlanSpline.Result(success=False, message=f"Transform fail: {exc}")

        abs_waypoints = self._compute_spline_absolute_segment(
            start_pose, goal_base_pose, self.max_translation_step
        )
        if not abs_waypoints:
            goal_handle.abort()
            return PlanSpline.Result(success=False, message="No waypoints generated")

        if not self.arm_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("ArmControl action server not available")
            goal_handle.abort()
            return PlanSpline.Result(success=False, message="ArmControl unavailable")

        success = await self._execute_waypoints(abs_waypoints, goal_handle, fixed_orientation=start_pose.orientation)
        if success:
            goal_handle.succeed()
            return PlanSpline.Result(success=True, message="OK")
        goal_handle.abort()
        return PlanSpline.Result(success=False, message="Execution failed")

    def _get_current_pose(self) -> Pose | None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=Duration(seconds=1.0),
            )
        except Exception as exc:
            self.get_logger().error(f"TF lookup failed: {exc}")
            return None

        pose = Pose()
        pose.position.x = tf.transform.translation.x
        pose.position.y = tf.transform.translation.y
        pose.position.z = tf.transform.translation.z
        pose.orientation = tf.transform.rotation
        return pose

    async def _transform_pose(self, pose_st: PoseStamped, target_frame: str) -> PoseStamped:
        # tf2_geometry_msgs import above registers converters for PoseStamped
        if not self.tf_buffer.can_transform(
            target_frame,
            pose_st.header.frame_id,
            rclpy.time.Time(),
            timeout=Duration(seconds=1.0),
        ):
            raise RuntimeError(f"No transform from {pose_st.header.frame_id} to {target_frame}")
        return self.tf_buffer.transform(pose_st, target_frame, timeout=Duration(seconds=1.0))

    def _compute_spline_absolute_segment(
        self, start: Pose, goal: Pose, max_translation_step: float
    ):
        start_pos = np.array([start.position.x, start.position.y, start.position.z], dtype=float)
        goal_pos = np.array([goal.position.x, goal.position.y, goal.position.z], dtype=float)
        translation_distance = float(np.linalg.norm(goal_pos - start_pos))
        if translation_distance < 1e-6:
            return [], []

        q_fixed = [
            start.orientation.x,
            start.orientation.y,
            start.orientation.z,
            start.orientation.w,
        ]

        t_knots = np.array([0.0, translation_distance])
        spline = CubicSpline(t_knots, np.vstack((start_pos, goal_pos)), axis=0, bc_type="clamped")
        steps = int(np.ceil(translation_distance / max_translation_step)) if translation_distance > 1e-4 else 1
        t_samples = np.linspace(t_knots[0], t_knots[-1], steps + 1)
        interp_positions = spline(t_samples)

        absolute_poses: List[Pose] = []
        for pos in interp_positions:
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pos
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q_fixed
            absolute_poses.append(pose)

        return absolute_poses

    async def _execute_waypoints(self, abs_waypoints: List[Pose], goal_handle, fixed_orientation) -> bool:
        total = len(abs_waypoints)
        for idx, tgt_abs in enumerate(abs_waypoints):
            tgt_abs.orientation = fixed_orientation
            current_pose = self._get_current_pose()
            if current_pose is None:
                self.get_logger().error("Failed to get current pose during execution")
                return False
            rel_wp = self._relative_from_current(current_pose, tgt_abs)

            goal = ArmControl.Goal()
            goal.target_pose = rel_wp
            self.get_logger().info(
                f"Sending waypoint {idx + 1}/{total} "
                f"rel=({rel_wp.position.x:.3f}, {rel_wp.position.y:.3f}, {rel_wp.position.z:.3f})"
            )
            gh_future = self.arm_client.send_goal_async(goal)
            gh = await gh_future
            if not gh.accepted:
                self.get_logger().error("Waypoint rejected by ArmControl")
                return False
            res_future = gh.get_result_async()
            res = await res_future
            result = res.result
            if result.result_code != ArmControl.Result.SUCCESS:
                msg = result.result_message if hasattr(result, "result_message") else result.message
                self.get_logger().error(f"Waypoint failed: {msg}")
                return False

            if goal_handle is not None:
                feedback = PlanSpline.Feedback()
                feedback.progress = float(idx + 1) / float(total)
                goal_handle.publish_feedback(feedback)
        return True


    # ----- Raster scan (service) -----
    async def handle_scan_request(self, request, response):
        # Simple raster in the base frame around the provided center pose.
        spacing_along = max(request.spacing, 0.05)  # min 5 cm
        spacing_lines = max(request.line_spacing if request.line_spacing > 0.0 else spacing_along, 0.05)
        width = request.width if request.width > 0 else 0.50
        height = request.height if request.height > 0 else 0.80

        start_pose = self._get_current_pose()
        if start_pose is None or isinstance(start_pose, Exception):
            response.success = False
            response.message = "Cannot fetch current pose"
            return response

        try:
            center_st = request.center_pose
            if center_st.header.frame_id != self.base_frame:
                center_st = self.tf_buffer.transform(center_st, self.base_frame, timeout=Duration(seconds=1.0))
            center = center_st.pose
        except Exception as exc:
            self.get_logger().error(f"Scan transform failed: {exc}")
            response.success = False
            response.message = f"Transform failed: {exc}"
            return response

        abs_poses = self._generate_raster(center, width, height, spacing_along, spacing_lines, start_pose.orientation)
        if len(abs_poses) == 0:
            response.success = False
            response.message = "No scan poses generated"
            return response

        self._publish_path_marker(abs_poses)

        success = await self._execute_waypoints(abs_poses, goal_handle=None, fixed_orientation=start_pose.orientation)
        if success:
            response.success = True
            response.message = "Raster executed"
        else:
            response.success = False
            response.message = "Raster execution failed"
        return response

    def _generate_raster(self, center: Pose, width: float, height: float, spacing_along: float, spacing_lines: float, orientation) -> List[Pose]:
        half_w = width / 2.0
        half_h = height / 2.0

        y_coords = []
        y = -half_w
        while y <= half_w + 1e-6:
            y_coords.append(y)
            y += spacing_along
        z_coords = []
        z = half_h
        while z >= -half_h - 1e-6:
            z_coords.append(z)
            z -= spacing_lines

        # Build S-pattern endpoints (only endpoints, spline will smooth corners)
        endpoints: List[np.ndarray] = []
        for idx, z_off in enumerate(z_coords):
            line = list(y_coords) if idx % 2 == 0 else list(reversed(y_coords))
            start = np.array([center.position.x, center.position.y + line[0], center.position.z + z_off], dtype=float)
            end = np.array([center.position.x, center.position.y + line[-1], center.position.z + z_off], dtype=float)
            if idx == 0:
                endpoints.append(start)
            endpoints.append(end)
            if idx < len(z_coords) - 1:
                next_start = np.array([center.position.x, center.position.y + line[-1], center.position.z + z_coords[idx + 1]], dtype=float)
                endpoints.append(next_start)

        points = np.vstack(endpoints)
        distances = np.linalg.norm(np.diff(points, axis=0), axis=1)
        t_knots = np.insert(np.cumsum(distances), 0, 0.0)
        if t_knots[-1] < 1e-6:
            return []
        spline = CubicSpline(t_knots, points, axis=0, bc_type="clamped")

        spacing_along = max(spacing_along, 0.05)
        num_steps = max(1, int(np.ceil(t_knots[-1] / spacing_along)))
        t_samples = np.linspace(0.0, t_knots[-1], num_steps + 1)

        poses: List[Pose] = []
        last = None
        for t in t_samples:
            pos = spline(t)
            if last is not None and np.linalg.norm(pos - last) < spacing_along - 1e-4:
                continue
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation = orientation
            poses.append(p)
            last = pos
        return poses

    def _publish_path_marker(self, poses: List[Pose]):
        marker = Marker()
        marker.header.frame_id = self.base_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "spline_scan"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.01
        marker.color.r = 0.0
        marker.color.g = 0.8
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.pose.orientation.w = 1.0

        for p in poses:
            pt = Point()
            pt.x = p.position.x
            pt.y = p.position.y
            pt.z = p.position.z
            marker.points.append(pt)

        self.path_pub.publish(marker)
    #will changed later to use SLerp
    def _relative_from_current(self, current: Pose, target: Pose) -> Pose:
        prev_matrix = tf_transformations.quaternion_matrix((
            current.orientation.x,
            current.orientation.y,
            current.orientation.z,
            current.orientation.w,
        ))
        prev_matrix[0:3, 3] = [current.position.x, current.position.y, current.position.z]

        tgt_matrix = tf_transformations.quaternion_matrix((
            target.orientation.x,
            target.orientation.y,
            target.orientation.z,
            target.orientation.w,
        ))
        tgt_matrix[0:3, 3] = [target.position.x, target.position.y, target.position.z]

        rel_matrix = np.dot(np.linalg.inv(prev_matrix), tgt_matrix)
        rel_quat = tf_transformations.quaternion_from_matrix(rel_matrix)

        wp = Pose()
        wp.position.x = rel_matrix[0, 3]
        wp.position.y = rel_matrix[1, 3]
        wp.position.z = rel_matrix[2, 3]
        wp.orientation.x = rel_quat[0]
        wp.orientation.y = rel_quat[1]
        wp.orientation.z = rel_quat[2]
        wp.orientation.w = rel_quat[3]
        return wp


def main() -> None:
    rclpy.init()
    node = SplinePlanner()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
