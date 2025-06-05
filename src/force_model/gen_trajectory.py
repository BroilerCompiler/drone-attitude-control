import numpy as np
import casadi as ca
from params import ExperimentParameters
p = ExperimentParameters()


def gen_square_traj(nx, initial, length):

    xref = np.zeros((p.N, 3))
    side = int(p.N/4)
    xref = np.zeros((p.N, 3))
    side = int(p.N/4)
    for i in range(side):
        xref[i, 0] = initial[0] + length*(i/side)
        xref[i, 1] = initial[1]
        xref[i, 2] = initial[2]
        xref[i+side, 0] = initial[0] + length
        xref[i+side, 1] = initial[1] + length*(i/side)
        xref[i+side, 2] = initial[2]
        xref[i+2*side, 0] = initial[0] + length - length*(i/side)
        xref[i+2*side, 1] = initial[1] + length
        xref[i+2*side, 2] = initial[2]
        xref[i+3*side, 0] = initial[0]
        xref[i+3*side, 1] = initial[1] + length - length*(i/side)
        xref[i+3*side, 2] = initial[2]
        xref[i, 0] = initial[0] + length*(i/side)
        xref[i, 1] = initial[1]
        xref[i, 2] = initial[2]
        xref[i+side, 0] = initial[0] + length
        xref[i+side, 1] = initial[1] + length*(i/side)
        xref[i+side, 2] = initial[2]
        xref[i+2*side, 0] = initial[0] + length - length*(i/side)
        xref[i+2*side, 1] = initial[1] + length
        xref[i+2*side, 2] = initial[2]
        xref[i+3*side, 0] = initial[0]
        xref[i+3*side, 1] = initial[1] + length - length*(i/side)
        xref[i+3*side, 2] = initial[2]

    return xref


def gen_static_point_traj(nx, initial):
    N = p.N_horizon+p.N+1
    xref = np.zeros((N, nx))
    xref[:, 0] = np.ones(N)*initial[0]
    xref[:, 1] = np.ones(N)*initial[1]
    return xref


def gen_straight_traj(nx, initial, length):
    # TODO: generate docstrings

    # const jerk
    jerk = 6*length / p.T**3

    xref = np.zeros((p.N_horizon+p.N+1, nx))
    for i in range(p.N_horizon+p.N+1):
        xref[i, 0] = initial[0] + 1/6 * jerk * (i * p.dt)**3
        xref[i, 1] = initial[1]
        xref[i, 2] = 0.5 * jerk * (i * p.dt)**2
        xref[i, 3] = 0
        xref[i, 4] = jerk * i * p.dt
        xref[i, 5] = 0

    return xref


def gen_circle_traj(nx, center, radius) -> np.array:

    omega = 2*np.pi/p.T

    xref = np.zeros((p.N_horizon+p.N+1, nx))
    for i in range(p.N_horizon+p.N+1):
        # pos ref
        xref[i, 0] = center[0] + radius*ca.cos(omega*i/p.N*p.T)
        xref[i, 1] = center[1] + radius*ca.sin(omega*i/p.N*p.T)

        # velocity ref
        xref[i, 2] = -radius*ca.sin(omega*i/p.N*p.T)*omega
        xref[i, 3] = radius*ca.cos(omega*i/p.N*p.T)*omega

    return xref


# define main function for testing
if __name__ == '__main__':
    from store_results import create_plots
    # plot circle traj
    radius = 1
    center = np.array([0, 0])
    circle = gen_circle_traj(4, center, radius)
    # u
    omega = 2*np.pi/p.T
    uref = np.zeros((p.N, 2))
    for i in range(p.N):
        uref[i, 0] = radius * ca.sin(omega*i/p.N*p.T)*(omega)**3
        uref[i, 1] = - radius * ca.cos(omega*i/p.N*p.T)*(omega)**3
    create_plots(circle, None, uref, store_plots=False)

    # plot straigt line traj
    length = 1
    straight = gen_straight_traj(6, center, length)
    # u
    jerk = 6*length / p.T**3
    u_vec2 = np.zeros((p.N, 2))
    u_vec2[:, 0] = np.ones(p.N) * jerk
    create_plots(straight, None, u_vec2, store_plots=False)
