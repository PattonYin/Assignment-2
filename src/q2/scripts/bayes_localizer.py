#!/usr/bin/env python3
"""Implement Markov localization over discretized headings and (x,y) coordinates."""

import math
from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Pose2D, Twist
from numpy.typing import NDArray
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

from a2_common import FAST_QoS, GridInfo, build_gt_map


class BayesLocalizer(Node):
    """Discrete Bayes localizer over a fixed occupancy map.

    References:
        Markov localization is described in Chapter 7.2 of
            "Probabilistic Robotics" by Thrun, Burgard, and Fox (2006).

        Grid-based localization using a histogram filter is described in Chapter 8.2.
    """

    def __init__(self) -> None:
        """Initialize map, belief, prediction timer, and correction subscription."""
        super().__init__("bayes_localizer")

        self.declare_parameter("origin_x_m", -2.0)
        self.declare_parameter("origin_y_m", -2.0)
        self.declare_parameter("resolution_m", 0.05)
        self.declare_parameter("height_cells", 80)
        self.declare_parameter("width_cells", 80)
        self.declare_parameter("frame_id", "map")

        self.declare_parameter("theta_bins", 36)
        self.declare_parameter("every_nth_beam", 3)
        self.declare_parameter("off_by_one_prob", 0.08)
        self.declare_parameter("prediction_rate_hz", 20.0)
        self.declare_parameter("scene_path", "")

        self._grid_info = GridInfo(
            origin_x=float(self.get_parameter("origin_x_m").value),
            origin_y=float(self.get_parameter("origin_y_m").value),
            resolution_m=float(self.get_parameter("resolution_m").value),
            height_cells=int(self.get_parameter("height_cells").value),
            width_cells=int(self.get_parameter("width_cells").value),
            parent_frame_id=str(self.get_parameter("frame_id").value),
        )
        self._theta_bins = int(self.get_parameter("theta_bins").value)
        """Number of angular discretization bins in the histogram filter."""

        self._every_nth_beam = int(self.get_parameter("every_nth_beam").value)
        self._off_by_one_prob = float(self.get_parameter("off_by_one_prob").value)

        scene_path_param = str(self.get_parameter("scene_path").value)
        if scene_path_param:
            scene_path = Path(scene_path_param)
        else:
            q1_share = Path(get_package_share_directory("q1"))
            scene_path = q1_share / "models" / "turtlebot_scene.xml"

        occ_bool = build_gt_map(grid_info=self._grid_info, scene_xml=scene_path)
        self._free_mask = ~occ_bool
        """Boolean mask over the ground-truth occupancy map where free cells are True."""

        self._theta_vals_rad = np.linspace(
            -np.pi, np.pi, self._theta_bins, endpoint=False, dtype=np.float32
        )
        """Array of angle values (radians) for each theta bin in the discretized grid."""

        self._theta_step_rad = math.tau / self._theta_bins
        """Angle (radians) between bins in the discretized grid."""

        self._belief = np.zeros(
            (
                self._theta_bins,
                self._grid_info.height_cells,
                self._grid_info.width_cells,
            ),
            dtype=np.float32,
        )
        """Discrete belief cells indexed by (theta bin, cell row, cell column)."""

        free_count = np.sum(self._free_mask)
        if not free_count:
            raise RuntimeError("No free cells found in occupancy map.")
        self._belief[:, self._free_mask] = 1.0 / (self._theta_bins * free_count)

        self._latest_vx_mps = 0.0
        self._latest_wz_radps = 0.0
        self._correction_step_count = 0

        self.create_subscription(
            LaserScan, "/laser_scans", self._laser_scan_cb, FAST_QoS
        )
        self.create_subscription(Twist, "/cmd_vel", self._cmd_vel_cb, FAST_QoS)
        self._pose_pub = self.create_publisher(Pose2D, "/estimated_odometry", 10)

        prediction_rate_hz = float(self.get_parameter("prediction_rate_hz").value)
        self._prediction_dt_s = 1.0 / prediction_rate_hz
        self.create_timer(self._prediction_dt_s, self._prediction_timer_cb)

    def _laser_scan_cb(self, msg: LaserScan) -> None:
        """Apply a correction update using the latest scan and publish the resulting MAP pose."""
        self._belief = self._correct_belief(
            self._belief, msg, every_nth_beam=self._every_nth_beam
        )

        pose_msg = self._map_pose_from_belief(self._belief)
        self._pose_pub.publish(pose_msg)

        self._correction_step_count += 1
        self.get_logger().info(
            f"correction={self._correction_step_count} "
            f"est_pose=(x={pose_msg.x:.3f}, y={pose_msg.y:.3f}, theta={pose_msg.theta:.3f})"
        )

    def _cmd_vel_cb(self, msg: Twist) -> None:
        """Store the latest velocity command for prediction updates."""
        self._latest_vx_mps = float(msg.linear.x)
        self._latest_wz_radps = float(msg.angular.z)

    def _prediction_timer_cb(self) -> None:
        """Run one prediction step using the latest velocity command."""
        self._belief = self._predict_belief(
            belief=self._belief,
            vx_mps=self._latest_vx_mps,
            wz_radps=self._latest_wz_radps,
            dt_s=self._prediction_dt_s,
            off_by_one_prob=self._off_by_one_prob,
        )

    def _map_pose_from_belief(self, belief: NDArray[np.float32]) -> Pose2D:
        """Convert the maximum a posteriori belief index to a world-frame 2D pose."""
        flat_idx = int(np.argmax(belief))
        theta_idx, y_cell, x_cell = np.unravel_index(flat_idx, belief.shape)

        pose_msg = Pose2D()
        pose_msg.x = self._grid_info.col_to_x(int(x_cell))
        pose_msg.y = self._grid_info.row_to_y(int(y_cell))
        pose_msg.theta = float(self._theta_vals_rad[theta_idx])

        return pose_msg

    def _normalize(self, belief: NDArray[np.float32]) -> NDArray[np.float32]:
        """Normalize a belief tensor to sum to one."""
        total_mass = np.sum(belief)
        if total_mass <= 1e-15:  # Only distribute uniform mass over free cells
            free_count = int(np.sum(self._free_mask))
            uniform_free = np.zeros_like(belief)
            uniform_free[:, self._free_mask] = 1.0 / (self._theta_bins * free_count)
            return uniform_free
        return belief / total_mass

    @staticmethod
    def _shift_no_wrap(grid: NDArray, dr_cells: int, dc_cells: int) -> NDArray:
        """Translate a 2D array without wrap-around.

        :param grid: 2D input grid
        :param dr_cells: Translation delta over grid rows
        :param dc_cells: Translation delta over grid columns
        :return: Shifted grid with out-of-bounds entries dropped
        """
        n_rows, n_cols = grid.shape
        out = np.zeros_like(grid)

        src_r0 = max(0, -dr_cells)
        src_r1 = min(n_rows, n_rows - dr_cells)
        dst_r0 = max(0, dr_cells)
        dst_r1 = dst_r0 + max(0, src_r1 - src_r0)

        src_c0 = max(0, -dc_cells)
        src_c1 = min(n_cols, n_cols - dc_cells)
        dst_c0 = max(0, dc_cells)
        dst_c1 = dst_c0 + max(0, src_c1 - src_c0)

        if dst_r0 >= dst_r1 or dst_c0 >= dst_c1:
            return out

        out[dst_r0:dst_r1, dst_c0:dst_c1] = grid[src_r0:src_r1, src_c0:src_c1]
        return out

    def _predict_belief(
        self,
        belief: NDArray[np.float32],
        vx_mps: float,
        wz_radps: float,
        dt_s: float,
        off_by_one_prob: float,
    ) -> NDArray[np.float32]:
        """Apply the Bayes filter prediction step over the discretized pose grid.

        Requirements for your implementation:
        - Integrate one velocity command over `dt_s` using the idealized velocity model
          from Eq. 5.9 (Chapter 5.3) of "Probabilistic Robotics" (Thrun et al., 2006).
            - You can import your motion model implementation using:

            from a2_common.motion_models import simulate_velocity_delta

        - Convert the predicted continuous displacement to row/column shifts and apply
          them independently for each theta bin. Use bilinear interpolation to
          distribute the fractional-cell remainder between neighboring cells.
        - Shift orientation bins according to angular displacement over the timestep, and
          again apply interpolation to distribute fractional remainders between bins.
        - Model angular uncertainty by distributing `off_by_one_prob` probability mass
          to neighboring theta bins (split evenly across +/- one-bin errors).
        - Mask out probability mass in known-occupied cells.
        - Return a normalized belief tensor.

        You may find the following utility functions or fields useful (use is optional):
        - `self._shift_no_wrap(grid, dr_cells, dc_cells)`
        - `self._normalize(belief)`
        - `self._theta_bins`, `self._theta_vals_rad`, and `self._theta_step_rad`
        - `self._grid_info.resolution_m`
        - `np.roll(..., axis=0)` for theta-bin wrap-around

        Reference: Markov grid localization prediction (Table 8.1, Chapter 8.1)
            using an idealized velocity-based motion model (Eq. 5.9, Chapter 5.3)
            from "Probabilistic Robotics" by Thrun, Burgard, and Fox (2006).

        :param belief: Prior belief tensor indexed by (theta bin, row, col)
        :param vx_mps: Commanded linear velocity in meters per second
        :param wz_radps: Commanded angular velocity in radians per second
        :param dt_s: Prediction timestep in seconds
        :param off_by_one_prob: Probability of off-by-one theta-bin transition noise
        :return: Predicted normalized belief tensor
        """
        return belief  # TODO

    def _correct_belief(
        self,
        belief: NDArray[np.float32],
        scan: LaserScan,
        every_nth_beam: int,
        hit_likelihood: float = 0.87,
        miss_likelihood: float = 0.24,
    ) -> NDArray[np.float32]:
        """Apply the Bayes filter correction step using the given laser scan.

        Requirements for your implementation:
        - Use every Nth beam in the scan (`every_nth_beam`) while preserving each
          selected beam's original index-to-angle correspondence.
        - Use only valid range measurements (finite and within sensor-reported limits).
        - For each theta bin and selected beam, compute the expected beam endpoint
          for each candidate (x, y) belief grid cell.
        - Compare the projected endpoint occupancy against the ground-truth map to
          compute per-cell beam likelihoods: when a predicted beam endpoint matches expected
          occupancy, consider the beam to have likelihood `hit_likelihood`. Otherwise,
          consider the beam to have likelihood `miss_likelihood`.
        - Assume that beams with out-of-map endpoints have likelihood `miss_likelihood`.
        - Mask out probability mass in known-occupied cells.
        - Return a normalized belief tensor.

        You may find the following utility functions or fields useful (use is optional):
        - `np.indices(...)`
        - `self._free_mask`
        - `self._grid_info` and its members `height_cells`, `width_cells`, and `resolution_m`

        Reference: Markov grid localization correction (Table 8.1, Chapter 8.1) using
            a simplified beam measurement model (Table 6.1, Chapter 6.3) adapted from
            "Probabilistic Robotics" by Thrun, Burgard, and Fox (2006).

        :param belief: Predicted belief tensor indexed by (theta bin, row, col)
        :param scan: Latest laser scan used for the correction update
        :param every_nth_beam: Beam subsampling stride (e.g., 1 means "use every beam")
        :param hit_likelihood: Likelihood used when projected endpoint occupancy matches
        :param miss_likelihood: Likelihood used when projected endpoint occupancy mismatches
        :return: Corrected normalized belief tensor
        """
        return belief  # TODO


def main() -> None:
    """Initialize ROS and spin the Q2 localizer node."""
    rclpy.init()
    node = BayesLocalizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
