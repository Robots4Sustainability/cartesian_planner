#!/usr/bin/env python3

from __future__ import annotations

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
    quaternion_matrix,
    quaternion_from_matrix,
)
from scipy.spatial.transform import Rotation as R, Slerp


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

        self.base_frame = self.get_parameter("base_frame").get_parameter_value().string_value
        self.ee_frame = self.get_parameter("ee_frame").get_parameter_value().string_value

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
        start_pos = np.array([start.position.x, start.position.y, start.position.z], dtype=float)
        goal_pos = np.array([goal.position.x, goal.position.y, goal.position.z], dtype=float)
        translation_distance = float(np.linalg.norm(goal_pos - start_pos))

        start_rot = R.from_quat([start.orientation.x, start.orientation.y, start.orientation.z, start.orientation.w])
        goal_rot = R.from_quat([goal.orientation.x, goal.orientation.y, goal.orientation.z, goal.orientation.w])
        angle = float((start_rot.inv() * goal_rot).magnitude())

        translation_steps = int(np.ceil(translation_distance / max_translation_step)) if translation_distance > 1e-4 else 0
        rotation_steps = int(np.ceil(angle / max_rotation_step)) if angle > 1e-4 else 0
        steps = max(translation_steps, rotation_steps, 1)

        ts = np.linspace(0.0, 1.0, steps + 1, dtype=float)[1:]
        slerp = Slerp([0.0, 1.0], R.from_quat([start_rot.as_quat(), goal_rot.as_quat()]))
        interp_rots = slerp(ts)
        interp_positions = start_pos + np.outer(ts, (goal_pos - start_pos))

        absolute_poses: List[Pose] = []
        for pos, quat in zip(interp_positions, interp_rots.as_quat()):
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pos
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = quat
            absolute_poses.append(pose)

        return self._absolute_to_relative(start, absolute_poses)

    # ------------------------------------------------------------------
    # Helper methods for constrained path generation
    # ------------------------------------------------------------------

    def _build_absolute_path(self, start: Pose, goal: Pose) -> List[Pose]:
        if self._pose_distance(start, goal) <= 1e-4:
            return []
        return [goal]

    def _pose_distance(self, a: Pose, b: Pose) -> float:
        vec_a = np.array([a.position.x, a.position.y, a.position.z], dtype=float)
        vec_b = np.array([b.position.x, b.position.y, b.position.z], dtype=float)
        return float(np.linalg.norm(vec_a - vec_b))

    def _absolute_to_relative(self, start_pose: Pose, absolute: List[Pose]) -> List[Pose]:
        if not absolute:
            return []

        rel_waypoints: List[Pose] = []
        prev_pose = start_pose
        for pose in absolute:
            prev_matrix = quaternion_matrix((
                prev_pose.orientation.x,
                prev_pose.orientation.y,
                prev_pose.orientation.z,
                prev_pose.orientation.w,
            ))
            prev_matrix[0:3, 3] = [prev_pose.position.x, prev_pose.position.y, prev_pose.position.z]

            current_matrix = quaternion_matrix((
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ))
            current_matrix[0:3, 3] = [pose.position.x, pose.position.y, pose.position.z]

            rel_matrix = np.dot(np.linalg.inv(prev_matrix), current_matrix)
            rel_quat = quaternion_from_matrix(rel_matrix)

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
