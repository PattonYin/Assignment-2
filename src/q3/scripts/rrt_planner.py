#!/usr/bin/env python3
"""Q3 RRT planner implementation file."""

from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Path as PathMsg
from nav_msgs.srv import GetPlan
from numpy.typing import NDArray
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from a2_common import (
    GridCell,
    GridInfo,
    build_gt_map,
    inflate_costmap,
    quat_msg_to_yaw_rad,
    unpack_occupancy_grid_msg,
    yaw_to_quaternion_msg,
)


def _wrap_angle_rad(theta_rad: float) -> float:
    """Wrap an angle within `[-pi, pi]`."""
    return (theta_rad + math.pi) % (2.0 * math.pi) - math.pi


class Pose2D(NamedTuple):
    """A planar pose with position `(x, y)` and heading `theta_rad`."""

    x: float
    y: float
    theta_rad: float


@dataclass(frozen=True)
class RRTNode:
    """A node in the RRT search tree."""

    pose: Pose2D
    parent_idx: int
    """Index of this node's parent within the RRT (-1 for root node)."""


class RRTPlanner(Node):
    """Single-tree goal-biased RRT planner service."""

    def __init__(self) -> None:
        """Initialize the ground-truth map, planner params, and service endpoints."""
        super().__init__("rrt_planner")

        self.declare_parameter("origin_x_m", -2.0)
        self.declare_parameter("origin_y_m", -2.0)
        self.declare_parameter("resolution_m", 0.05)
        self.declare_parameter("height_cells", 80)
        self.declare_parameter("width_cells", 80)
        self.declare_parameter("occupied_threshold", 85)
        self.declare_parameter("robot_radius_m", 0.078)
        self.declare_parameter("goal_bias", 0.15)
        self.declare_parameter("max_iters", 20000)
        self.declare_parameter("step_size_m", 0.25)
        self.declare_parameter("goal_tolerance_m", 0.2)
        self.declare_parameter("goal_heading_tolerance_rad", 0.5)
        self.declare_parameter("collision_step_m", 0.05)
        self.declare_parameter("control_dt_s", 0.2)
        self.declare_parameter("turn_wz_radps", 1.2)
        self.declare_parameter("heading_distance_weight", 0.2)
        self.declare_parameter("rng_tag", "q3-planner")
        self.declare_parameter("scene_path", "")

        self._grid_info = GridInfo(
            origin_x=float(self.get_parameter("origin_x_m").value),
            origin_y=float(self.get_parameter("origin_y_m").value),
            resolution_m=float(self.get_parameter("resolution_m").value),
            height_cells=int(self.get_parameter("height_cells").value),
            width_cells=int(self.get_parameter("width_cells").value),
            parent_frame_id="map",
        )
        self._occupied_threshold = int(self.get_parameter("occupied_threshold").value)
        self._robot_radius_m = float(self.get_parameter("robot_radius_m").value)

        self._goal_bias = float(self.get_parameter("goal_bias").value)
        self._max_iters = int(self.get_parameter("max_iters").value)
        self._step_size_m = float(self.get_parameter("step_size_m").value)
        self._goal_tolerance_m = float(self.get_parameter("goal_tolerance_m").value)
        self._goal_heading_tolerance_rad = float(
            self.get_parameter("goal_heading_tolerance_rad").value
        )
        self._collision_step_m = float(self.get_parameter("collision_step_m").value)
        self._control_dt_s = float(self.get_parameter("control_dt_s").value)
        self._turn_wz_radps = float(self.get_parameter("turn_wz_radps").value)
        self._heading_distance_weight = float(
            self.get_parameter("heading_distance_weight").value
        )
        self._control_vx_mps = self._step_size_m / max(self._control_dt_s, 1e-6)

        self._control_options = [
            (self._control_vx_mps, 0.0),
            (self._control_vx_mps, self._turn_wz_radps),
            (self._control_vx_mps, -self._turn_wz_radps),
        ]

        rng_tag = str(self.get_parameter("rng_tag").value)
        self._rng = np.random.default_rng(self._seed_from_tag(rng_tag))

        self._costmap_0_100 = self._load_default_costmap()

        self.create_subscription(
            OccupancyGrid, "/gt/occupancy_map", self._occupancy_grid_cb, 10
        )
        self._path_pub = self.create_publisher(PathMsg, "/planned_path", 10)

        self._tree_edges_xy: list[tuple[float, float, float, float]] = []
        """A list of tree edges represented as (start_x, start_y, end_x, end_y) tuples.

        This list is used only for RRT visualization. It's meant to be useful, but is not required.
        """

        self._tree_publish_stride_iters = 50
        self._rrt_tree_pub = self.create_publisher(MarkerArray, "/rrt_tree_markers", 1)
        self.create_timer(0.5, self._publish_rrt_tree_cb)

        self.create_service(GetPlan, "/plan_path", self._plan_path_cb)

    def _load_default_costmap(self) -> NDArray[np.int16]:
        """Load and inflate the default occupancy map from the assignment scene."""
        scene_path_param = str(self.get_parameter("scene_path").value)
        if scene_path_param:
            scene_path = Path(scene_path_param)
        else:
            q1_share = Path(get_package_share_directory("q1"))
            scene_path = q1_share / "models" / "turtlebot_scene.xml"

        occ_bool = build_gt_map(grid_info=self._grid_info, scene_xml=scene_path)
        return inflate_costmap(
            occ_grid=occ_bool.astype(np.int16) * 100,
            grid_info=self._grid_info,
            occupied_threshold=self._occupied_threshold,
            inflation_radius_m=self._robot_radius_m,
        )

    def _occupancy_grid_cb(self, msg: OccupancyGrid) -> None:
        """Update the planner costmap from an external occupancy grid topic."""
        grid_data, grid_info = unpack_occupancy_grid_msg(msg)
        self._costmap_0_100 = np.clip(grid_data.astype(np.int16), 0, 100)
        self._grid_info = grid_info

    def _plan_path_cb(
        self, req: GetPlan.Request, res: GetPlan.Response
    ) -> GetPlan.Response:
        """Handle a path planning service call.

        :param req: Planning request containing start and goal poses
        :param res: Service response to populate with the planned path
        :return: Populated planning response
        """
        start_pose = Pose2D(
            x=float(req.start.pose.position.x),
            y=float(req.start.pose.position.y),
            theta_rad=quat_msg_to_yaw_rad(req.start.pose.orientation),
        )
        goal_pose = Pose2D(
            x=float(req.goal.pose.position.x),
            y=float(req.goal.pose.position.y),
            theta_rad=quat_msg_to_yaw_rad(req.goal.pose.orientation),
        )

        poses = self._generate_rrt(start_pose=start_pose, goal_pose=goal_pose)
        path_msg = self._build_path_msg(poses)

        res.plan = path_msg
        self._path_pub.publish(path_msg)
        return res

    def _generate_rrt(self, start_pose: Pose2D, goal_pose: Pose2D) -> list[Pose2D]:
        """Generate an RRT and return the root-to-goal pose path.

        :param start_pose: Start pose in map frame
        :param goal_pose: Goal pose in map frame
        :return: Pose waypoints from start to goal; empty if planning fails
        """
        self._tree_edges_xy = []  # Used for RRT visualization in RViz

        # TODO: Implement the rest of this method

        # Initialize the RRT node list using the start pose as the root node
        tree_nodes: list[RRTNode] = []  # TODO

        for _ in range(self._max_iters):
            # With probability `self._goal_bias`, use the goal pose as the random state,
            #   otherwise sample a random state using `self._random_state`

            # Find the nearest RRT node to the random state using `self._nearest_neighbor_idx`

            # Select a control input to steer toward the random state from the nearest node
            #   and compute the resulting new state (`self._select_input_and_new_state`)
            # If `_select_input_and_new_state` returns None, `continue` to skip this iteration

            # Once nearest_node and new_pose are defined, uncomment to store the edge for visualization:
            # self._tree_edges_xy.append(
            #     (nearest_node.pose.x, nearest_node.pose.y, new_pose.x, new_pose.y)
            # )
            # if len(self._tree_edges_xy) % self._tree_publish_stride_iters == 0:
            #     self._publish_rrt_tree_cb()

            # Create a node for the new state that resulted from the selected control input

            # If the new node is sufficiently close to the goal pose (`self._is_goal_reached`),
            #   extract the solution path from the list of tree nodes (`self._extract_path`)
            pass

        self.get_logger().info(f"Reached {self._max_iters} RRT iterations; exiting...")
        return []

    def _random_state(self, num_attempts: int = 100) -> Pose2D | None:
        """Sample a random collision-free state within the costmap bounds.

        :param num_attempts: Maximum number of times to attempt resampling (default: 100)
        :return: Random free pose in the map, or None if sampling fails
        """
        return None  # TODO

    @staticmethod
    def _rho_distance(pose_a: Pose2D, pose_b: Pose2D, heading_weight: float) -> float:
        """Compute the state-space metric `rho` between two planar poses.

        Scale the angular error (abs. rad) by `heading_weight` when summing with distance error (m).

        :param pose_a: First pose used to compute the distance
        :param pose_b: Second pose used to compute the distance
        :param heading_weight: Weight value used for angular error
        :return: Computed distance metric
        """
        return 0.0  # TODO

    def _nearest_neighbor_idx(
        self, tree_nodes: list[RRTNode], random_pose: Pose2D
    ) -> int:
        """Find the index of the nearest tree node to the given query pose.

        :param tree_nodes: List of nodes in the RRT
        :param random_pose: Sampled random 2D pose
        :return: Index of the nearest tree node, according to the rho state-space metric
        """
        return 0  # TODO

    def _new_state_trajectory(
        self,
        start_pose: Pose2D,
        vx_mps: float,
        wz_radps: float,
        duration_s: float,
        collision_check_step_m: float,
    ) -> list[Pose2D]:
        """Discretize `NEW_STATE(x, u, dt)` into collision-checkable poses.

        Rather than simulate one large motion step, which could skip over thin
        obstacles, simulate many steps spaced according to `collision_check_step_m`.

        You may wish to use the `simulate_velocity_command` function importable
        from `a2_common.motion_models` to simulate robot motion. This function
        is derived from the motion model you implemented in Question 2.

        :param start_pose: Initial state
        :param vx_mps: Linear velocity command in meters per second
        :param wz_radps: Angular velocity command in radians per second
        :param duration_s: Overall rollout duration in seconds
        :param collision_check_step_m: Spacing (meters) between poses checked for collisions
        :return: Sequence of propagated states (excluding the initial state)
        """
        return []  # TODO

    def _select_input_and_new_state(
        self,
        nearest_node: RRTNode,
        random_pose: Pose2D,
        control_options: list[tuple[float, float]],
    ) -> Pose2D | None:
        """Select and simulate the velocity command that best steers toward the random pose.

        The "best" control option is the one that results in a collision-free trajectory
            ending in the state that best minimizes the `self._rho_distance` metric.

        If no control option produces a collision-free trajectory, return None.

        :param nearest_node: Nearest node in the RRT to the target random pose
        :param random_pose: Random pose sampled within the costmap
        :param control_options: List of possible (vx_mps, wz_radps) velocity commands
        :return: New pose resulting from the best control option, or None if all produce collisions
        """
        return None  # TODO

    def _in_collision(self, x: float, y: float) -> bool:
        """Check whether the robot is in collision at the given point.

        Because obstacles in the stored costmap have already been inflated by the radius
        of the robot, a single collision check at the robot's center is sufficient.

        If the given point is outside the costmap, return True (i.e., treat it as in collision).

        :param x: Map-frame x-coordinate of the robot
        :param y: Map-frame y-coordinate of the robot
        :return: True if the point is in collision, otherwise False
        """
        return True  # TODO

    def _world_to_cell(self, x: float, y: float) -> GridCell | None:
        """Convert world coordinates to costmap cell indices (None if cell is invalid)."""
        grid_cell = self._grid_info.coord_to_cell((x, y))
        if not self._grid_info.is_valid_cell(grid_cell):
            return None
        return grid_cell

    @staticmethod
    def _extract_path(tree_nodes: list[RRTNode], goal_idx: int) -> list[Pose2D]:
        """Compute a path by backtracking through parents from the goal node to the root.

        :param tree_nodes: Nodes in the RRT
        :param goal_idx: Index of the goal node
        :return: List of states in the path (root to goal)
        """
        points_rev = []
        idx = goal_idx
        while idx >= 0:
            node = tree_nodes[idx]
            points_rev.append(node.pose)
            idx = node.parent_idx
        points_rev.reverse()
        return points_rev

    def _is_goal_reached(self, pose: Pose2D, goal_pose: Pose2D) -> bool:
        """Check whether a pose satisfies position and heading goal tolerances."""
        pos_ok = (
            math.hypot(goal_pose.x - pose.x, goal_pose.y - pose.y)
            <= self._goal_tolerance_m
        )
        heading_ok = (
            abs(_wrap_angle_rad(goal_pose.theta_rad - pose.theta_rad))
            <= self._goal_heading_tolerance_rad
        )
        return pos_ok and heading_ok

    def _build_path_msg(self, states: list[Pose2D]) -> PathMsg:
        """Build a nav_msgs/Path from a sequence of planar states."""
        path_msg = PathMsg()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"

        for state in states:
            pose_msg = PoseStamped()
            pose_msg.header = path_msg.header
            pose_msg.pose.position.x = state.x
            pose_msg.pose.position.y = state.y
            pose_msg.pose.orientation = yaw_to_quaternion_msg(state.theta_rad)
            path_msg.poses.append(pose_msg)

        return path_msg

    def _publish_rrt_tree_cb(self) -> None:
        """Publish the currently cached RRT tree as a line-list marker."""
        marker_msg = Marker()
        marker_msg.header.frame_id = "map"
        marker_msg.header.stamp = self.get_clock().now().to_msg()
        marker_msg.ns = "rrt_tree"
        marker_msg.id = 0
        marker_msg.action = Marker.DELETE if not self._tree_edges_xy else Marker.ADD
        if self._tree_edges_xy:
            marker_msg.type = Marker.LINE_LIST
            marker_msg.pose.orientation.w = 1.0
            marker_msg.scale.x = 0.01
            marker_msg.color.r = 0.1
            marker_msg.color.g = 0.8
            marker_msg.color.b = 1.0
            marker_msg.color.a = 0.8
            for x0, y0, x1, y1 in self._tree_edges_xy:
                marker_msg.points.extend(
                    (Point(x=x0, y=y0, z=0.0), Point(x=x1, y=y1, z=0.0))
                )
        marker_array_msg = MarkerArray()
        marker_array_msg.markers.append(marker_msg)
        self._rrt_tree_pub.publish(marker_array_msg)

    @staticmethod
    def _seed_from_tag(seed_tag: str) -> int:
        """Return a stable uint32 seed derived from a user-provided tag."""
        return zlib.crc32(seed_tag.encode("utf-8")) & 0xFFFFFFFF


def main() -> None:
    """Initialize ROS and spin the Q3 planner node."""
    rclpy.init()
    node = RRTPlanner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
