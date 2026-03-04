"""Construct an occupancy grid map based on laser scans received from the robot."""

import numpy as np
from numpy.typing import NDArray

from a2_common import GridInfo, PosedLaserScan


class OccupancyGrid:
    """A 2D occupancy grid using log odds to represent the probability of occupancy."""

    def __init__(self, grid_info: GridInfo, min_obstacle_depth_m: float) -> None:
        """Initialize the occupancy grid.

        :param grid_info: Defines the origin, resolution, height, and width of the grid
        :param min_obstacle_depth_m: Minimum depth (m) assumed for obstacles when ray-tracing
        """
        self.grid_info = grid_info
        self.min_obstacle_depth_m = min_obstacle_depth_m

        # Log-odds representation: L = log( p(occupied) / p(free) )
        # Initialize to log( 0.5 / 0.5 ) = log(1) = 0 (equal probability of occupied and free)
        self.log_odds = np.zeros(
            (grid_info.height_cells, grid_info.width_cells), dtype=np.float32
        )

    @staticmethod
    def prob_to_log_odds(
        prob: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Convert a scalar probability or array of probability values into log odds.

        Reference: Eq. 9.5 (Chapter 9.2) of "Probabilistic Robotics" by Thrun, Burgard, and Fox (2006).
        """
        return np.zeros_like(prob)  # TODO

    @staticmethod
    def log_odds_to_prob(
        log_odds: float | NDArray[np.floating],
    ) -> float | NDArray[np.floating]:
        """Convert a scalar or array of log odds values into probability values.

        Reference: Eq. 9.6 (Chapter 9.2) of "Probabilistic Robotics" by Thrun, Burgard, and Fox (2006).
        """
        return np.zeros_like(log_odds)  # TODO

    def update(
        self, scan: PosedLaserScan, *, p_free: float = 0.2, p_occupied: float = 0.8
    ) -> None:
        """Update the occupancy grid using an inverse sensor model based on ray tracing.

        Requirements for your implementation:
        - Integrate one laser scan captured at a known pose (`scan.sensor_pose`) into `self.log_odds`.
        - Use only valid range measurements (i.e., finite and within sensor limits).
            Message structure: https://docs.ros.org/en/jazzy/p/sensor_msgs/msg/LaserScan.html
        - Preserve the correspondence between kept range measurements and their original beam angles.
        - Convert each valid beam from the sensor frame to the map frame, then update
          traversed cells as evidence of free space and obstacle-hit cells as occupied space.
        - Account for obstacle thickness using `self.min_obstacle_depth_m`, representing the
          assumed minimum depth (in meters) of any obstacle in the environment.
        - Skip updates to out-of-bounds grid cells.
        - Updates must accumulate over time (do not reset the map state).

        Beam endpoint treatment: Treat grid cells from the sensor to the beam
            endpoint cell as free space, excluding the endpoint cell itself.

        Obstacle depth treatment: Treat grid cells from the beam endpoint through
            the minimum obstacle depth as occupied, including the endpoint cell.

        You may find the following utility functions useful (use is optional):
        - `self.grid_info.x_to_col(...)` and `self.grid_info.y_to_row(...)`
        - `self.grid_info.coord_to_cell((x, y))`
        - `self.grid_info.is_valid_cell(cell)`
        - `bresenham_line(start_cell, end_cell)` (importable from `a2_common`)

        By convention, the occupancy grid origin is placed at its bottom-left corner
            on the xy-plane. Rows increase as y increases; columns increase as x increases.

        Reference: Table 9.1 (Chapter 9.2) of "Probabilistic Robotics" by Thrun, Burgard,
            and Fox (2006). Figure 9.3 visualizes the inverse sensor model described above.

        :param scan: Laser scan from a known pose, to be incorporated into the grid
        :param p_free: Probability that a cell is occupied given a laser passes through it
        :param p_occupied: Probability that a cell is occupied given a laser hits in it
        """
        pass  # TODO
