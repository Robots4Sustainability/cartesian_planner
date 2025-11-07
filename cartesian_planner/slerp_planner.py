#!/usr/bin/env python3
"""Standalone SLERP-based Cartesian planner node."""

from __future__ import annotations

import math
from typing import List

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from geometry_msgs.msg import Pose, PoseStamped, TransformStamped
from cartesian_planner.srv import PlanCartesianPath

import tf2_ros
import tf2_geometry_msgs  # noqa: F401
import numpy as np
from tf_transformations import (
    quaternion_slerp,
    quaternion_matrix,
    quaternion_from_matrix,
)


def pose_to_tf(pose: Pose) -> tuple:
    return (
        pose.position.x,
        pose.position.y,
        pose.position.z,
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )


class SlerpPlanner(Node):
    def __init__(self) -> None:
        super().__init__("slerp_planner")

        self.declare_parameter("base_frame", "eddie_base_link")
        self.declare_parameter("ee_frame", "eddie_right_arm_robotiq_85_grasp_link")
        self.declare_parameter("safe_hover_z", 0.8)
        self.declare_parameter("torso_clear_radius", 0.32)
        self.declare_parameter("radius_clearance", 0.05)

        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.ee_frame = self.get_parameter("ee_frame").get_parameter_value().string_value
        self.safe_hover_z = float(self.get_parameter("safe_hover_z").value)
        self.torso_clear_radius = float(self.get_parameter("torso_clear_radius").value)
        self.radius_clearance = float(self.get_parameter("radius_clearance").value)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.service = self.create_service(
            PlanCartesianPath,
            "plan_cartesian_path",
            self.plan_callback,
        )

        self.get_logger().info(f"SLERP planner ready. base_frame={self.base_frame} ee_frame={self.ee_frame}")

    def plan_callback(self, request: PlanCartesianPath.Request, response: PlanCartesianPath.Response):
        max_trans = max(request.max_translation_step, 1e-4)
        max_rot = max(request.max_rotation_step, 1e-4)

        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.ee_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
        except tf2_ros.LookupException as exc:
            response.success = False
            response.message = f"TF lookup failed: {exc}"
            return response
        except tf2_ros.ExtrapolationException as exc:
            response.success = False
            response.message = f"TF extrapolation failed: {exc}"
            return response

        start_pose = Pose()
        start_pose.position.x = tf.transform.translation.x
        start_pose.position.y = tf.transform.translation.y
        start_pose.position.z = tf.transform.translation.z
        start_pose.orientation = tf.transform.rotation

        goal_pose = request.target_pose.pose

        self.get_logger().info(
            f"Received plan request Δpos({goal_pose.position.x - start_pose.position.x:.3f}, "
            f"{goal_pose.position.y - start_pose.position.y:.3f}, "
            f"{goal_pose.position.z - start_pose.position.z:.3f})"
        )

        waypoints = self._plan_with_constraints(start_pose, goal_pose, max_trans, max_rot)

        self.get_logger().info(f"Generated {len(waypoints)} relative waypoints")

        response.success = True
        response.relative_waypoints = waypoints
        response.message = f"Generated {len(waypoints)} relative waypoints."
        return response

    def _plan_with_constraints(
        self,
        start: Pose,
        goal: Pose,
        max_translation_step: float,
        max_rotation_step: float,
    ) -> List[Pose]:
        absolute_waypoints = self._build_absolute_path(start, goal)

        waypoints: List[Pose] = []
        current = start
        for target in absolute_waypoints:
            segment = self._compute_relative_segment(current, target, max_translation_step, max_rotation_step)
            waypoints.extend(segment)
            current = target
        self.get_logger().info("correctly generating")
        return waypoints

    def _compute_relative_segment(
        self,
        start: Pose,
        goal: Pose,
        max_translation_step: float,
        max_rotation_step: float,
    ) -> List[Pose]:
        sx, sy, sz, qsx, qsy, qsz, qsw = pose_to_tf(start)
        gx, gy, gz, qgx, qgy, qgz, qgw = pose_to_tf(goal)

        translation_distance = math.sqrt((gx - sx) ** 2 + (gy - sy) ** 2 + (gz - sz) ** 2)

        def normalize(qx: float, qy: float, qz: float, qw: float) -> tuple:
            norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
            if norm < 1e-12:
                return (0.0, 0.0, 0.0, 1.0)
            return (qx / norm, qy / norm, qz / norm, qw / norm)

        qsx, qsy, qsz, qsw = normalize(qsx, qsy, qsz, qsw)
        qgx, qgy, qgz, qgw = normalize(qgx, qgy, qgz, qgw)

        dot = qsx * qgx + qsy * qgy + qsz * qgz + qsw * qgw
        angle = 2.0 * math.acos(min(1.0, max(-1.0, abs(dot))))

        translation_steps = int(math.ceil(translation_distance / max_translation_step)) if translation_distance > 1e-4 else 0
        rotation_steps = int(math.ceil(angle / max_rotation_step)) if angle > 1e-4 else 0
        steps = max(translation_steps, rotation_steps, 1)

        waypoints: List[Pose] = []

        prev_pos = (sx, sy, sz)
        prev_quat = (qsx, qsy, qsz, qsw)

        for i in range(steps):
            t = float(i + 1) / float(steps)
            interp_pos = (
                sx + (gx - sx) * t,
                sy + (gy - sy) * t,
                sz + (gz - sz) * t,
            )
            interp_quat = quaternion_slerp(
                (qsx, qsy, qsz, qsw),
                (qgx, qgy, qgz, qgw),
                t,
            )

            rel_pos = (
                interp_pos[0] - prev_pos[0],
                interp_pos[1] - prev_pos[1],
                interp_pos[2] - prev_pos[2],
            )

            prev_matrix = quaternion_matrix(prev_quat)
            prev_matrix[0:3, 3] = prev_pos

            current_matrix = quaternion_matrix(interp_quat)
            current_matrix[0:3, 3] = interp_pos

            rel_matrix = np.dot(np.linalg.inv(prev_matrix), current_matrix)

            rel_pos = rel_matrix[0:3, 3]
            rel_quat = quaternion_from_matrix(rel_matrix)

            waypoint = Pose()
            waypoint.position.x = rel_pos[0]
            waypoint.position.y = rel_pos[1]
            waypoint.position.z = rel_pos[2]
            waypoint.orientation.x = rel_quat[0]
            waypoint.orientation.y = rel_quat[1]
            waypoint.orientation.z = rel_quat[2]
            waypoint.orientation.w = rel_quat[3]
            waypoints.append(waypoint)

            prev_pos = interp_pos
            prev_quat = interp_quat

        return waypoints

    # ------------------------------------------------------------------
    # Helper methods for constrained path generation
    # ------------------------------------------------------------------

    def _build_absolute_path(self, start: Pose, goal: Pose) -> List[Pose]:
        eps = 1e-4
        hover_z = max(self.safe_hover_z, start.position.z, goal.position.z)
        path: List[Pose] = []

        # Helper to avoid duplicate consecutive poses
        def append_pose(p: Pose):
            if not path:
                path.append(p)
                return
            last = path[-1]
            if self._pose_distance(last, p) > eps:
                path.append(p)

        current = start

        if abs(current.position.z - hover_z) > eps:
            append_pose(self._make_pose(current.position.x, current.position.y, hover_z, current.orientation))
            current = path[-1]

        safe_start_xy = self._adjust_xy(current.position.x, current.position.y)
        if abs(safe_start_xy[0] - current.position.x) > eps or abs(safe_start_xy[1] - current.position.y) > eps:
            append_pose(self._make_pose(safe_start_xy[0], safe_start_xy[1], hover_z, current.orientation))
            current = path[-1]

        safe_goal_xy = self._adjust_xy(goal.position.x, goal.position.y)
        append_pose(self._make_pose(safe_goal_xy[0], safe_goal_xy[1], hover_z, goal.orientation))
        current = path[-1]

        if abs(goal.position.z - hover_z) > eps:
            append_pose(self._make_pose(goal.position.x, goal.position.y, hover_z, goal.orientation))

        append_pose(goal)

        return path

    def _make_pose(self, x: float, y: float, z: float, orientation) -> Pose:
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = z
        pose.orientation.x = orientation.x
        pose.orientation.y = orientation.y
        pose.orientation.z = orientation.z
        pose.orientation.w = orientation.w
        return pose

    def _pose_distance(self, a: Pose, b: Pose) -> float:
        dx = a.position.x - b.position.x
        dy = a.position.y - b.position.y
        dz = a.position.z - b.position.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _adjust_xy(self, x: float, y: float) -> tuple:
        min_radius = self.torso_clear_radius + self.radius_clearance
        r = math.sqrt(x * x + y * y)
        if r >= min_radius:
            return x, y

        if r < 1e-4:
            angle = -math.pi / 2.0  # default to pointing forward (negative Y)
        else:
            angle = math.atan2(y, x)

        x = math.cos(angle) * min_radius
        y = math.sin(angle) * min_radius
        return x, y


def main() -> None:
    rclpy.init()
    node = None

    try:
        node = SlerpPlanner()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        if node is not None:
            node.get_logger().info("SLERP planner interrupted by user.")
    except Exception as e:

        if node is not None:
            node.get_logger().error(f"Unexpected exception: {e}")
        raise
    finally:

        if node is not None:
            node.get_logger().info("Destroying SLERP planner node...")
            node.destroy_node()


        if rclpy.ok():
            rclpy.shutdown()
        else:

            try:
                rclpy.shutdown()
            except Exception:
                pass


        import time
        time.sleep(1.0)



if __name__ == "__main__":
    main()
