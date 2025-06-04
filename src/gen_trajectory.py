import numpy as np
import casadi as ca
from matplotlib import pyplot as plt
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


def compare_reftraj_vs_sim(t, reftraj, simX, u):
    _, ax0 = plt.subplots(2)
    fig1, ax1 = plt.subplots(3, sharex=True)
    # z over x
    if simX is not None:
        ax0[0].plot(simX[:, 0], simX[:, 1], label='$p^\\mathrm{sim}$')
        ax0[1].plot(simX[:, 2], simX[:, 3], label='$v^\\mathrm{sim}$')
    if reftraj is not None:
        ax0[0].plot(reftraj[:, 0], reftraj[:, 1], label='$p^\\mathrm{ref}$')
        ax0[1].plot(reftraj[:, 2], reftraj[:, 3], label='$v^\\mathrm{ref}$')

    # component-wise
    if simX is not None:
        ax1[0].plot(t, simX[:, 0], label='$p^\\mathrm{sim}_\\mathrm{x}$')
        ax1[1].plot(t, simX[:, 2], label='$v^\\mathrm{sim}_\\mathrm{x}$')
        ax1[0].plot(t, simX[:, 1], label='$p^\\mathrm{sim}_\\mathrm{z}$')
        ax1[1].plot(t, simX[:, 3], label='$v^\\mathrm{sim}_\\mathrm{z}$')
    if reftraj is not None:
        ax1[0].plot(t, reftraj[:, 0], label='$p^\\mathrm{ref}_\\mathrm{x}$')
        ax1[1].plot(t, reftraj[:, 2], label='$v^\\mathrm{ref}_\\mathrm{x}$')
        ax1[0].plot(t, reftraj[:, 1], label='$p^\\mathrm{ref}_\\mathrm{z}$')
        ax1[1].plot(t, reftraj[:, 3], label='$v^\\mathrm{ref}_\\mathrm{z}$')
    if u is not None:
        ax1[2].plot(t[:-1], u[:, 0], label='$\\theta$')
        ax1[2].plot(t[:-1], u[:, 1], label='$F_\\mathrm{d}$')
    fig1.supxlabel('Seconds')
    ax0[0].legend()
    ax0[1].legend()
    ax1[0].legend()
    ax1[1].legend()
    ax1[2].legend()
    ax0[0].grid()
    ax0[1].grid()
    ax1[0].grid()
    ax1[1].grid()
    ax1[2].grid()
    ax0[0].axis('equal')
    ax0[1].axis('equal')

    plt.show()


# define main function for testing
if __name__ == '__main__':
    # plot circle traj
    radius = 1
    center = np.array([0, 0])
    circle = gen_circle_traj(6, center, radius)
    # u
    omega = 2*np.pi/p.T
    uref = np.zeros((p.N, 2))
    for i in range(p.N):
        uref[i, 0] = radius * ca.sin(omega*i/p.N*p.T)*(omega)**3
        uref[i, 1] = - radius * ca.cos(omega*i/p.N*p.T)*(omega)**3
    compare_reftraj_vs_sim(t=np.linspace(0, p.T, p.N+1),
                           reftraj=circle[:p.N+1], simX=None, u=uref)

    # plot straigt line traj
    length = 1
    straight = gen_straight_traj(6, center, length)
    # u
    jerk = 6*length / p.T**3
    u_vec2 = np.zeros((p.N, 2))
    u_vec2[:, 0] = np.ones(p.N) * jerk
    compare_reftraj_vs_sim(t=np.linspace(0, p.T, p.N+1),
                           reftraj=straight[:p.N+1], simX=None, u=u_vec2)
