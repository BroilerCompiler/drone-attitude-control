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

        # acceleration ref
        xref[i, 4] = - radius * ca.cos(omega*i/p.N*p.T)*(omega)**2
        xref[i, 5] = - radius * ca.sin(omega*i/p.N*p.T)*(omega)**2

    return xref


def compare_reftraj_vs_sim(t, reftraj, simX, u):
    _, ax0 = plt.subplots(2, 2)
    fig1, ax1 = plt.subplots(4, sharex=True)
    # z over x
    if simX is not None:
        ax0[0, 0].plot(simX[:, 0], simX[:, 1], label='p_sim')
        ax0[0, 1].plot(simX[:, 2], simX[:, 3], label='v_sim')
        ax0[1, 0].plot(simX[:, 4], simX[:, 5], label='a_sim')
    if reftraj is not None:
        ax0[0, 0].plot(reftraj[:, 0], reftraj[:, 1], label='p_ref')
        ax0[0, 1].plot(reftraj[:, 2], reftraj[:, 3], label='v_ref')
        ax0[1, 0].plot(reftraj[:, 4], reftraj[:, 5], label='a_ref')
    if u is not None:
        ax0[1, 1].plot(u[:, 0], u[:, 1], label='h')

    # component-wise
    if simX is not None:
        ax1[0].plot(t, simX[:, 0], label='p_sim_x')
        ax1[1].plot(t, simX[:, 2], label='v_sim_x')
        ax1[2].plot(t, simX[:, 4], label='a_sim_x')
        ax1[0].plot(t, simX[:, 1], label='p_sim_z')
        ax1[1].plot(t, simX[:, 3], label='v_sim_z')
        ax1[2].plot(t, simX[:, 5], label='a_sim_z')
    if reftraj is not None:
        ax1[0].plot(t, reftraj[:, 0], label='p_ref_x')
        ax1[1].plot(t, reftraj[:, 2], label='v_ref_x')
        ax1[2].plot(t, reftraj[:, 4], label='a_ref_x')
        ax1[0].plot(t, reftraj[:, 1], label='p_ref_z')
        ax1[1].plot(t, reftraj[:, 3], label='v_ref_z')
        ax1[2].plot(t, reftraj[:, 5], label='a_ref_z')
    if u is not None:
        ax1[3].plot(t[:-1], u[:, 0], label='h_x')
        ax1[3].plot(t[:-1], u[:, 1], label='h_z')
    fig1.supxlabel('Seconds')
    ax0[0, 0].legend()
    ax0[0, 1].legend()
    ax0[1, 0].legend()
    ax0[1, 1].legend()
    ax1[0].legend()
    ax1[1].legend()
    ax1[2].legend()
    ax1[3].legend()
    ax0[0, 0].grid()
    ax0[0, 1].grid()
    ax0[1, 0].grid()
    ax0[1, 1].grid()
    ax1[0].grid()
    ax1[1].grid()
    ax1[2].grid()
    ax1[3].grid()
    ax0[0, 0].axis('equal')
    ax0[0, 1].axis('equal')
    ax0[1, 0].axis('equal')
    ax0[1, 1].axis('equal')

    plt.show()


# define main function for testing
if __name__ == '__main__':
    # plot circle traj
    radius = 1
    center = np.array([0, 0])
    circle = gen_circle_traj(6, center, radius)
    # u
    omega = 2*np.pi/p.T
    u_vec = np.zeros((p.N, 2))
    for i in range(p.N):
        u_vec[i, 0] = radius * ca.sin(omega*i/p.N*p.T)*(omega)**3
        u_vec[i, 1] = - radius * ca.cos(omega*i/p.N*p.T)*(omega)**3
    compare_reftraj_vs_sim(t=np.linspace(0, p.T, p.N+1),
                           reftraj=circle[:p.N+1], simX=None, u=u_vec)

    # plot straigt line traj
    length = 1
    straight = gen_straight_traj(6, center, length)
    # u
    jerk = 6*length / p.T**3
    u_vec2 = np.zeros((p.N, 2))
    u_vec2[:, 0] = np.ones(p.N) * jerk
    compare_reftraj_vs_sim(t=np.linspace(0, p.T, p.N+1),
                           reftraj=straight[:p.N+1], simX=None, u=u_vec2)
