#!/usr/bin/env python3
"""
Spline planner action server:
- Accepts a goal pose expressed in the end-effector frame (relative move).
- Looks up current EE pose, transforms goal to base frame, builds a cubic spline,
  converts absolute samples to relative deltas, and sends them to ArmControl sequentially.
- No visualization to keep it minimal for service use by pick_place.
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
import tf2_geometry_msgs  # registers Pose/PoseStamped type support for tf2
from cartesian_planner.action import PlanSpline
from eddie_ros.action import ArmControl
from scipy.interpolate import CubicSpline
import tf_transformations


class SplinePlanner(Node):
    def __init__(self) -> None:
        super().__init__("spline_planner")
        self.declare_parameter("base_frame", "eddie_base_link")
        self.declare_parameter("ee_frame", "eddie_right_arm_end_effector_link")
        self.declare_parameter("arm_action_server", "right_arm/arm_control")
        self.declare_parameter("max_translation_step", 0.05)

        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.arm_action_server = self.get_parameter("arm_action_server").value
        self.max_translation_step = float(self.get_parameter("max_translation_step").value)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.arm_client = ActionClient(self, ArmControl, self.arm_action_server)

        self.action_server = ActionServer(
            self,
            PlanSpline,
            "spline_plan",
            execute_callback=self.execute_callback,
        )

        self.get_logger().info(
            f"Spline planner ready. base_frame={self.base_frame}, ee_frame={self.ee_frame}, arm_server={self.arm_action_server}"
        )

    async def execute_callback(self, goal_handle):
        goal: PlanSpline.Goal = goal_handle.request
        self.get_logger().info("Received spline goal (EE frame).")

        start_pose = await self._get_current_pose()
        if start_pose is None:
            goal_handle.abort()
            return PlanSpline.Result(success=False, message="Cannot fetch current pose")

        try:
            goal_st = PoseStamped()
            goal_st.header.frame_id = self.ee_frame
            # Using time=0 (latest available) to avoid future extrapolation errors
            goal_st.header.stamp = rclpy.time.Time().to_msg()
            goal_st.pose = goal.target_pose
            goal_in_base: PoseStamped = await self._transform_pose(goal_st, self.base_frame)
            goal_base_pose = goal_in_base.pose
        except Exception as exc:
            self.get_logger().error(f"Failed to transform goal to base: {exc}")
            goal_handle.abort()
            return PlanSpline.Result(success=False, message=f"Transform fail: {exc}")

        rel_waypoints, _ = self._compute_spline_relative_segment(
            start_pose, goal_base_pose, self.max_translation_step
        )
        if not rel_waypoints:
            goal_handle.abort()
            return PlanSpline.Result(success=False, message="No waypoints generated")

        if not self.arm_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().error("ArmControl action server not available")
            goal_handle.abort()
            return PlanSpline.Result(success=False, message="ArmControl unavailable")

        success = await self._execute_waypoints(rel_waypoints, goal_handle)
        if success:
            goal_handle.succeed()
            return PlanSpline.Result(success=True, message="OK")
        goal_handle.abort()
        return PlanSpline.Result(success=False, message="Execution failed")

    async def _get_current_pose(self) -> Pose | None:
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
        # Explicitly check availability to avoid extrapolation errors
        if not self.tf_buffer.can_transform(
            target_frame,
            pose_st.header.frame_id,
            rclpy.time.Time(),
            timeout=Duration(seconds=1.0),
        ):
            raise RuntimeError(f"No transform from {pose_st.header.frame_id} to {target_frame}")
        return self.tf_buffer.transform(pose_st, target_frame, timeout=Duration(seconds=1.0))

    def _compute_spline_relative_segment(
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

        relative_segment = self._absolute_to_relative(start, absolute_poses)
        return relative_segment, absolute_poses

    async def _execute_waypoints(self, waypoints: List[Pose], goal_handle) -> bool:
        total = len(waypoints)
        for idx, wp in enumerate(waypoints):
            goal = ArmControl.Goal()
            goal.target_pose = wp
            self.get_logger().info(
                f"Sending waypoint {idx + 1}/{total} "
                f"rel=({wp.position.x:.3f}, {wp.position.y:.3f}, {wp.position.z:.3f})"
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

            feedback = PlanSpline.Feedback()
            feedback.progress = float(idx + 1) / float(total)
            goal_handle.publish_feedback(feedback)
        return True

    def _absolute_to_relative(self, start_pose: Pose, absolute_poses: List[Pose]) -> List[Pose]:
        if not absolute_poses:
            return []

        rel_waypoints: List[Pose] = []
        prev_pose = start_pose
        for pose in absolute_poses:
            prev_matrix = tf_transformations.quaternion_matrix((
                prev_pose.orientation.x,
                prev_pose.orientation.y,
                prev_pose.orientation.z,
                prev_pose.orientation.w,
            ))
            prev_matrix[0:3, 3] = [prev_pose.position.x, prev_pose.position.y, prev_pose.position.z]

            current_matrix = tf_transformations.quaternion_matrix((
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ))
            current_matrix[0:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

            rel_matrix = np.dot(np.linalg.inv(prev_matrix), current_matrix)
            rel_quat = tf_transformations.quaternion_from_matrix(rel_matrix)

            waypoint = Pose()
            waypoint.position.x = rel_matrix[0, 3]
            waypoint.position.y = rel_matrix[1, 3]
            waypoint.position.z = rel_matrix[2, 3]
            waypoint.orientation.x = rel_quat[0]
            waypoint.orientation.y = rel_quat[1]
            waypoint.orientation.z = rel_quat[2]
            waypoint.orientation.w = rel_quat[3]
            rel_waypoints.append(waypoint)

            prev_pose = pose

        return rel_waypoints


def main() -> None:
    rclpy.init()
    node = SplinePlanner()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
