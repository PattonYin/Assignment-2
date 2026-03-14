"""Define motion-model utility functions."""
import math

def simulate_velocity_delta(
    theta_rad: float,
    vx_mps: float,
    wz_radps: float,
    dt_s: float,
    eps_wz_radps: float = 1e-6,
) -> tuple[float, float, float]:
    """Compute the nominal change in robot pose using an idealized velocity motion model.

    Avoid numerical instability by using a straight-line motion model when the
        absolute commanded angular velocity is near zero (based on `eps_wz_radps`).

    Reference: Idealized velocity motion model given by Equation 5.9 from
        Chapter 5.3 in "Probabilistic Robotics" by Thrun, Burgard, and Fox (2006).

    :param theta_rad: Robot heading in radians
    :param vx_mps: Commanded linear velocity in meters per second
    :param wz_radps: Commanded angular velocity in radians per second
    :param dt_s: Simulation timestep in seconds
    :param eps_wz_radps: Threshold for treating angular velocity as zero (default: 1e-6)
    :return: Delta in robot pose as the tuple `(dx_m, dy_m, dtheta_rad)`
    """
    theta = theta_rad
    v = vx_mps
    w = wz_radps
    dt = dt_s

    dtheta = w * dt
    if abs(w) < eps_wz_radps:
        dx = v * dt * math.cos(theta)
        dy = v * dt * math.sin(theta)
    else:
        dx = -v / w * math.sin(theta) + v / w * math.sin(theta + dtheta)
        dy = v / w * (math.cos(theta) - math.cos(theta + dtheta))

    return dx, dy, dtheta


def simulate_velocity_command(
    x: float,
    y: float,
    theta_rad: float,
    vx_mps: float,
    wz_radps: float,
    dt_s: float,
    eps_wz_radps: float = 1e-6,
) -> tuple[float, float, float]:
    """Simulate planar robot motion using an idealized velocity motion model.

    Reference: Idealized velocity motion model given by Equation 5.9 from
        Chapter 5.3 in "Probabilistic Robotics" by Thrun, Burgard, and Fox (2006).

    :param x: Current robot base pose x-coordinate
    :param y: Current robot base pose y-coordinate
    :param theta_rad: Current robot heading in radians
    :param vx_mps: Commanded linear velocity in meters per second
    :param wz_radps: Commanded angular velocity in radians per second
    :param dt_s: Simulation timestep in seconds
    :param eps_wz_radps: Threshold for treating angular velocity as zero (default: 1e-6)
    :return: Resulting robot pose tuple `(x, y, theta_rad)`
    """
    dx_m, dy_m, dtheta_rad = simulate_velocity_delta(
        theta_rad=theta_rad,
        vx_mps=vx_mps,
        wz_radps=wz_radps,
        dt_s=dt_s,
        eps_wz_radps=eps_wz_radps,
    )
    return x + dx_m, y + dy_m, theta_rad + dtheta_rad
