import numpy as np
import casadi as ca
from params import ExperimentParameters, DroneData
dd = DroneData()
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


def gen_static_point_traj(nx, initial) -> np.array:
    N = p.N_horizon+p.N+1
    # sample traj at the converter frequency (higher than MPC frequency)
    N = N*p.ctrls_per_sample
    xref = np.zeros((N, nx))
    xref[:, 0] = np.ones(N)*initial[0]
    xref[:, 1] = np.ones(N)*initial[1]

    return xref


def gen_straight_traj(nx, initial, length) -> np.array:
    # TODO: generate docstrings

    N = p.N_horizon+p.N+1
    # sample traj at the converter frequency (higher than MPC frequency)
    N = N*p.ctrls_per_sample

    # const jerk
    jerk = 6*length / p.T**3

    xref = np.zeros((N, nx))
    for i in range(N):
        xref[i, 0] = initial[0] + 1/6 * jerk * (i * p.dt_converter)**3
        xref[i, 1] = initial[1] + 1/6 * jerk * (i * p.dt_converter)**3
        xref[i, 2] = 0.5 * jerk * (i * p.dt_converter)**2
        xref[i, 3] = 0.5 * jerk * (i * p.dt_converter)**2

    return xref


def gen_circle_traj(nx, center, radius) -> np.array:
    # sample traj at the converter frequency (higher than MPC frequency)
    N = p.N*p.ctrls_per_sample

    xref = np.zeros((N+p.N_horizon*p.ctrls_per_sample, nx))

    omega = 2*np.pi/p.T
    i = np.linspace(0, p.T, N)

    xref[:N, 0] = center[0] + radius * np.cos(omega*i)
    xref[:N, 1] = center[1] + radius * np.sin(omega*i)
    xref[:N, 2] = -radius * omega * np.sin(omega*i)
    xref[:N, 3] = radius * omega * np.cos(omega*i)
    if nx > 4:
        xref[:N, 4] = -radius * omega**2 * np.cos(omega*i)
        xref[:N, 5] = -radius * omega**2 * np.sin(omega*i)

    xref[N:] = xref[:p.N_horizon*p.ctrls_per_sample]

    return xref


def gen_straight_u(nu, length):
    N = p.N + p.N_horizon
    uref_ctrl = np.zeros((N, nu))
    jerk = 6*length / p.T**3
    uref_ctrl[:, 0] = np.ones(N) * jerk  # h_x
    uref_ctrl[:, 1] = np.ones(N) * jerk  # h_z
    return uref_ctrl


def gen_circle_u(nu, radius):
    N = p.N + p.N_horizon
    uref_ctrl = np.zeros((N, nu))

    i = np.linspace(0, p.T, N)

    omega = 2*np.pi/p.T
    uref_ctrl[:, 0] = radius * omega**3 * np.sin(omega*i)
    uref_ctrl[:, 1] = -radius * omega**3 * np.cos(omega*i)

    return uref_ctrl


def gen_reference_u(nu):
    # generate trajectory with hover thrust as reference
    uref = np.zeros((p.N+p.N_horizon, nu))
    # uref[:, 1] = np.ones(uref.shape[0]) * dd.GRAVITY

    return uref


# define main function for testing
if __name__ == '__main__':
    from store_results import create_plots
    # plot circle traj
    circle = gen_circle_traj(6, [0, 0], radius=1)
    uref = gen_circle_u(2, radius=1)
    create_plots(circle, None, uref, store_plots=False)

    # plot straigt line traj
    straight = gen_straight_traj(6, [0, 0], length=1)
    uref = gen_straight_u(2, length=1)
    create_plots(straight, None, uref, store_plots=False)
