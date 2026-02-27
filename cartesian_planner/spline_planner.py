#!/usr/bin/env python3
"""
Planner with action server:
- Accepts a goal pose expressed in the end-effector frame (relative move).
- Looks up current EE pose, transforms goal to base frame, generates a straight-line path,
  converts absolute samples to relative deltas, and sends them to ArmControl sequentially.
- Performs a raster scan when triggered with a service call and publishes a path marker.
"""

from typing import List

import asyncio
import json
import random
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.duration import Duration
from geometry_msgs.msg import Pose, PoseStamped
from tf2_ros import Buffer, TransformListener
from cartesian_planner.srv import PlanScanPath
from eddie_ros.action import ArmControl
from scipy.interpolate import interp1d
import tf_transformations
import tf2_geometry_msgs  # Registers PoseStamped transforms for tf2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
import time 

class SplinePlanner(Node):
    def __init__(self) -> None:
        super().__init__("spline_planner")
        self.declare_parameter("base_frame", "eddie_base_link")
        self.declare_parameter("ee_frame", "eddie_right_arm_end_effector_link")
        self.declare_parameter("arm_action_server", "right_arm/arm_control")

        self.cb_group = ReentrantCallbackGroup()
        self.base_frame = self.get_parameter("base_frame").value
        self.ee_frame = self.get_parameter("ee_frame").value
        self.arm_action_server = self.get_parameter("arm_action_server").value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.arm_client = ActionClient(self, ArmControl, self.arm_action_server, callback_group=self.cb_group)
        self.path_pub = self.create_publisher(Marker, "/spline_scan_path", 10)
        self.detected_objects: dict = {}

        self.scan_service = self.create_service(PlanScanPath, "plan_scan_path", self.handle_scan_request, callback_group=self.cb_group)

        self.get_logger().info(
            f"Planner ready. base_frame={self.base_frame}, ee_frame={self.ee_frame}, arm_server={self.arm_action_server}"
        )

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

    async def _execute_waypoints(self, abs_waypoints: List[Pose], fixed_orientation) -> bool:
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
            await self.precieve_objects()
        return True

    async def precieve_objects(self) -> None:
        time.sleep(3.0)
        if random.random() < 0.75:
            self.get_logger().info("No objects found")
            return

        frame_id = "eddie_right_arm_camera_link"
        class_name = random.choice(("speaker", "ecu"))
        detection = {
            "class": class_name,
            "confidence": round(random.uniform(0.60, 0.99), 3),
            "pose": {
                "frame_id": frame_id,
                "position": {
                    "x": round(random.uniform(0.25, 0.75), 3),
                    "y": round(random.uniform(-0.30, 0.30), 3),
                    "z": round(random.uniform(-0.15, 0.45), 3),
                },
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            },
        }

        previous = self.detected_objects.get(class_name)
        if previous is None or detection["confidence"] > previous["confidence"]:
            self.detected_objects[class_name] = detection
            self.get_logger().info(f"Updated {class_name} (mock): {detection}")
        else:
            self.get_logger().info(
                f"Skipped {class_name} (mock): new confidence {detection['confidence']} <= stored {previous['confidence']}"
            )


    # ----- Raster scan (service) -----
    async def handle_scan_request(self, request, response):
        spacing_along = 0.10 # diff b/w each waypoint
        spacing_lines = 0.12 # diff b/w horizontal parallel lines
        self.detected_objects = {}

        start_pose = self._get_current_pose()
        if start_pose is None or isinstance(start_pose, Exception):
            response.success = False
            response.message = "Cannot fetch current pose"
            return response

        try:
            top_left = self._pose_in_base(request.top_left)
            top_right = self._pose_in_base(request.top_right)
            bottom_right = self._pose_in_base(request.bottom_right)
            bottom_left = self._pose_in_base(request.bottom_left)
        except Exception as exc:
            self.get_logger().error(f"Scan transform failed: {exc}")
            response.success = False
            response.message = f"Transform failed: {exc}"
            return response

        abs_poses = self._generate_raster_from_corners(
            top_left, top_right, bottom_right, bottom_left, spacing_along, spacing_lines, start_pose.orientation
        )
        if len(abs_poses) == 0:
            response.success = False
            response.message = "No scan poses generated"
            return response

        self._publish_path_marker(abs_poses)

        success = await self._execute_waypoints(abs_poses, fixed_orientation=start_pose.orientation)
        if success:
            response.success = True
            response.message = json.dumps(
                {"status": "Raster executed", "detected_objects": self.detected_objects}
            )
        else:
            response.success = False
            response.message = "Raster execution failed"
        return response

    def _generate_raster_from_corners(
        self,
        top_left: Pose,
        top_right: Pose,
        bottom_right: Pose,
        bottom_left: Pose,
        spacing_along: float,
        spacing_lines: float,
        orientation,
    ) -> List[Pose]:
        left_edge = np.array(
            [
                [top_left.position.x, top_left.position.y, top_left.position.z],
                [bottom_left.position.x, bottom_left.position.y, bottom_left.position.z],
            ],
            dtype=float,
        )
        right_edge = np.array(
            [
                [top_right.position.x, top_right.position.y, top_right.position.z],
                [bottom_right.position.x, bottom_right.position.y, bottom_right.position.z],
            ],
            dtype=float,
        )

        left_points = self._interpolate_edge_points(left_edge, spacing_lines)
        right_points = self._interpolate_edge_points(right_edge, spacing_lines)
        if not left_points or not right_points:
            return []

        row_count = min(len(left_points), len(right_points))
        knots: List[np.ndarray] = []
        for idx in range(row_count):
            if idx % 2 == 0:
                knots.append(left_points[idx])
                knots.append(right_points[idx])
            else:
                knots.append(right_points[idx])
                knots.append(left_points[idx])

        if row_count > 0 and not np.allclose(knots[-1], right_points[row_count - 1]):
            knots.append(right_points[row_count - 1])

        if len(knots) < 2:
            return []

        sampled = self._sample_polyline(np.vstack(knots), spacing_along)
        poses: List[Pose] = []
        for pos in sampled:
            p = Pose()
            p.position.x, p.position.y, p.position.z = pos
            p.orientation = orientation
            poses.append(p)
        return poses

    def _pose_in_base(self, pose_st: PoseStamped) -> Pose:
        if pose_st.header.frame_id and pose_st.header.frame_id != self.base_frame:
            pose_st = self.tf_buffer.transform(pose_st, self.base_frame, timeout=Duration(seconds=1.0))
        return pose_st.pose

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

    def _interpolate_edge_points(self, edge: np.ndarray, spacing: float) -> List[np.ndarray]:
        vec = edge[1] - edge[0]
        length = float(np.linalg.norm(vec))
        if length < 1e-9:
            return []
        count = max(1, int(np.ceil(length / spacing)))
        ts = np.linspace(0.0, 1.0, count + 1)
        fx = interp1d([0.0, 1.0], [edge[0][0], edge[1][0]], kind="linear")
        fy = interp1d([0.0, 1.0], [edge[0][1], edge[1][1]], kind="linear")
        fz = interp1d([0.0, 1.0], [edge[0][2], edge[1][2]], kind="linear")
        xs = fx(ts)
        ys = fy(ts)
        zs = fz(ts)
        return [np.array([x, y, z], dtype=float) for x, y, z in zip(xs, ys, zs)]

    def _sample_polyline(self, points: np.ndarray, spacing: float) -> np.ndarray:
        if points.shape[0] < 2:
            return points
        deltas = np.diff(points, axis=0)
        seg_lens = np.linalg.norm(deltas, axis=1)
        cum = np.insert(np.cumsum(seg_lens), 0, 0.0)
        total = float(cum[-1])
        if total < 1e-9:
            return points[:1]
        count = max(1, int(np.ceil(total / spacing)))
        samples = np.linspace(0.0, total, count + 1)
        fx = interp1d(cum, points[:, 0], kind="linear")
        fy = interp1d(cum, points[:, 1], kind="linear")
        fz = interp1d(cum, points[:, 2], kind="linear")
        xs = fx(samples)
        ys = fy(samples)
        zs = fz(samples)
        return np.vstack((xs, ys, zs)).T


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
