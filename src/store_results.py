from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from jerk_model.gen_trajectory import gen_circle_traj
from acados_template.plot_utils import latexify_plot
from params import DroneData, ExperimentParameters
dd = DroneData()
p = ExperimentParameters()


def store_data(XRef, XSim, UOpt):
    d = datetime.today().strftime('%Y-%m-%d %H_%M_%S')
    XRef_path, XSim_path, UOpt_path = f'../experiment_data/npy/{d}_XRef.npy', f'../experiment_data/npy/{d}_XSim.npy', f'../experiment_data/npy/{d}_UOpt.npy'
    np.save(XRef_path, XRef)
    np.save(XSim_path, XSim)
    np.save(UOpt_path, UOpt)
    print(f'Stored data to ../experiment_data/npy/')
    return XRef_path, XSim_path, UOpt_path


def create_position(XRef, XSim, d, store_plots, show_plots):
    width = 3.5  # inches
    height = 3.5  # inches
    ls_ref = ':'
    ls_sim = '-'
    col_x_1 = 'deepskyblue'
    col_x_2 = 'skyblue'

    # position z over x
    plt.figure(figsize=(width, height))
    if XSim is not None:
        plt.plot(XSim[:, 0], XSim[:, 1],
                 label='$p^\\mathrm{sim}$', linestyle=ls_sim, color=col_x_1, alpha=0.7)
        plt.hlines(y=[dd.min_p_x, dd.max_p_x], xmin=dd.min_p_z, xmax=dd.max_p_z, linewidth=1,
                   linestyles='--', color='black')
        plt.vlines(x=[dd.min_p_z, dd.max_p_z], ymin=dd.min_p_x, ymax=dd.max_p_x, linewidth=1,
                   linestyles='--', color='black')
        plt.scatter(XSim[0, 0], XSim[0, 1],
                    label='Starting point', color='green')
        plt.scatter(XSim[int(XSim.shape[0]/4), 0], XSim[int(XSim.shape[0]/4), 1],
                    label='direction', color='lightgreen')
    if XRef is not None:
        plt.plot(XRef[:, 0], XRef[:, 1],
                 label='$p^\\mathrm{ref}$', linestyle=ls_ref, color=col_x_2, alpha=0.7)
        plt.scatter(XRef[int(XRef.shape[0]/4), 0], XRef[int(XRef.shape[0]/4), 1],
                    label='direction ref', color='lightblue')
    plt.axis('equal')
    plt.grid()
    plt.xlabel('Position (m)')
    plt.ylabel('Position (m)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if store_plots:
        plt.savefig('../experiment_data/img/' + d + '_traj_pos.pdf',
                    pad_inches=0, bbox_inches='tight')
    if show_plots:
        plt.show()


def create_velocity(XRef, XSim, d, store_plots, show_plots):
    width = 3.5  # inches
    height = 3.5  # inches
    ls_ref = ':'
    ls_sim = '-'
    col_x_1 = 'deepskyblue'
    col_x_2 = 'skyblue'

    # velocity z over x
    plt.figure(figsize=(width, height))
    if XSim is not None:
        plt.plot(XSim[:, 2], XSim[:, 3],
                 label='$v^\\mathrm{sim}$', linestyle=ls_sim, color=col_x_1, alpha=0.7)
        plt.hlines(y=[dd.min_v_x, dd.max_v_x], xmin=dd.min_v_z, xmax=dd.max_v_z, linewidth=1,
                   linestyles='--', color='black')
        plt.vlines(x=[dd.min_v_z, dd.max_v_z], ymin=dd.min_v_x, ymax=dd.max_v_x, linewidth=1,
                   linestyles='--', color='black')
        plt.scatter(XSim[0, 2], XSim[0, 3],
                    label='Starting point', color='green')
        plt.scatter(XSim[int(XSim.shape[0]/4), 2], XSim[int(XSim.shape[0]/4), 3],
                    label='direction', color='lightgreen')
    if XRef is not None:
        plt.plot(XRef[:, 2], XRef[:, 3],
                 label='$v^\\mathrm{ref}$', linestyle=ls_ref, color=col_x_2, alpha=0.7)
        plt.scatter(XRef[int(XRef.shape[0]/4), 2], XRef[int(XRef.shape[0]/4), 3],
                    label='direction ref', color='lightblue')
    plt.axis('equal')
    plt.grid()
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if store_plots:
        plt.savefig('../experiment_data/img/' + d + '_traj_vel.pdf',
                    pad_inches=0, bbox_inches='tight')
    if show_plots:
        plt.show()


def create_acceleration(XRef, a, d, store_plots, show_plots):
    width = 3.5  # inches
    height = 3.5  # inches
    ls_ref = ':'
    ls_sim = '-'
    col_x_1 = 'deepskyblue'
    col_x_2 = 'skyblue'

    # acceleration z over x
    plt.figure(figsize=(width, height))
    if a is not None:
        plt.plot(a[:, 0], a[:, 1],
                 label='$a^\\mathrm{sim}$', linestyle=ls_sim, color=col_x_1, alpha=0.7, marker='x')
        plt.hlines(y=[dd.min_a_x, dd.max_a_x], xmin=dd.min_a_z, xmax=dd.max_a_z, linewidth=1,
                   linestyles='--', color='black')
        plt.vlines(x=[dd.min_a_z, dd.max_a_z], ymin=dd.min_a_x, ymax=dd.max_a_x, linewidth=1,
                   linestyles='--', color='black')
        plt.scatter(a[0, 0], a[0, 1],
                    label='Starting point', color='green')

    if XRef is not None:
        plt.plot(XRef[:, 4], XRef[:, 5],
                 label='$a^\\mathrm{ref}$', linestyle=ls_ref, color=col_x_2, alpha=0.7)
        plt.scatter(XRef[int(XRef.shape[0]/4), 4], XRef[int(XRef.shape[0]/4), 5],
                    label='direction ref', color='lightblue')
    plt.axis('equal')
    plt.grid()
    plt.xlabel('Acceleration (m/s)')
    plt.ylabel('Acceleration (m/s)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if store_plots:
        plt.savefig('../experiment_data/img/' + d + '_traj_vel.pdf',
                    pad_inches=0, bbox_inches='tight')
    if show_plots:
        plt.show()


def create_componentwise(dt, XRef, XSim, a, UOpt, d, store_plots, show_plots):
    # component-wise
    ls_ref = ':'
    ls_sim = '-'
    col_x_1 = 'skyblue'
    col_x_2 = 'deepskyblue'
    col_u = 'darkgreen'
    width = 5  # inches
    height = 5  # inches

    fig, ax = plt.subplots(5, sharex=True, figsize=(
        width, height), layout='constrained')
    x = np.arange(0, p.T, dt)

    # component-wise
    if XSim is not None:
        ax[0].plot(x, XSim[:, 0], label='$p^\\mathrm{sim}_\\mathrm{x}$',
                   linestyle=ls_sim, color=col_x_1, alpha=0.7)
        ax[0].plot(x, XSim[:, 1], label='$p^\\mathrm{sim}_\\mathrm{z}$',
                   linestyle=ls_sim, color=col_x_1, alpha=0.7)
        ax[1].plot(x, XSim[:, 2], label='$v^\\mathrm{sim}_\\mathrm{x}$',
                   linestyle=ls_sim, color=col_x_1, alpha=0.7)
        ax[1].plot(x, XSim[:, 3], label='$v^\\mathrm{sim}_\\mathrm{z}$',
                   linestyle=ls_sim, color=col_x_1, alpha=0.7)
    if XRef is not None:
        ax[0].plot(x, XRef[:, 0],
                   label='$p^\\mathrm{ref}_\\mathrm{x}$', linestyle=ls_ref, color=col_x_2, alpha=0.7)
        ax[0].plot(x, XRef[:, 1],
                   label='$p^\\mathrm{ref}_\\mathrm{z}$', linestyle=ls_ref, color=col_x_2, alpha=0.7)
        ax[1].plot(x, XRef[:, 2],
                   label='$v^\\mathrm{ref}_\\mathrm{x}$', linestyle=ls_ref, color=col_x_2, alpha=0.7)
        ax[1].plot(x, XRef[:, 3],
                   label='$v^\\mathrm{ref}_\\mathrm{z}$', linestyle=ls_ref, color=col_x_2, alpha=0.7)
        ax[2].plot(x, XRef[:, 4],
                   label='$a^\\mathrm{ref}_\\mathrm{x}$', linestyle=ls_ref, color=col_x_2, alpha=0.7)
        ax[2].plot(x, XRef[:, 5],
                   label='$a^\\mathrm{ref}_\\mathrm{z}$', linestyle=ls_ref, color=col_x_2, alpha=0.7)
    if a is not None:
        ax[2].plot(
            x, a[:, 0], label='$a^\\mathrm{sim}_\\mathrm{x}$', linestyle=ls_sim, color=col_x_2, alpha=0.7)
        ax[2].plot(
            x, a[:, 1], label='$a^\\mathrm{sim}_\\mathrm{z}$', linestyle=ls_sim, color=col_x_2, alpha=0.7)
    if UOpt is not None:
        ax[3].plot(x, UOpt[:, 0], label='$\\theta$',
                   linestyle=ls_sim, color=col_u, alpha=0.7)
        ax[4].plot(x, UOpt[:, 1],
                   label='$F_\\mathrm{d}$', linestyle=ls_sim, color=col_u, alpha=0.7)
    fig.supxlabel('Time (s)')
    fig.align_ylabels()
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    ax[4].grid()
    ax[0].set_ylabel('Position (m)')
    ax[1].set_ylabel('Velocity (m/s)')
    ax[3].set_ylabel('Acceleration (m/s)')
    ax[3].set_ylabel('Angle (rad)')
    ax[4].set_ylabel('Thrust (N)')
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[4].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if store_plots:
        plt.savefig('../experiment_data/img/' + d + '_trajectory_component.pdf',
                    pad_inches=0, bbox_inches='tight')
    if show_plots:
        plt.show()


def create_plots(dt, XRef, XSim, a, UOpt, store_plots=False, show_plots=True):
    if not show_plots and not store_plots:
        return None

    print("Creating plots...")

    latexify_plot()
    d = datetime.today().strftime('%Y-%m-%d %H_%M_%S')
    create_position(XRef, XSim, d, store_plots, show_plots)
    create_velocity(XRef, XSim, d, store_plots, show_plots)
    create_acceleration(XRef, a, d, store_plots, show_plots)
    create_componentwise(dt, XRef, XSim, a, UOpt, d, store_plots, show_plots)

    if store_plots:
        print('Stored plots to ../experiment_data/img/')


def calc_aed(pref, psim):
    euclidean_distances = np.sqrt((pref - psim) ** 2)
    average_euclidean_distance = np.mean(euclidean_distances)
    return average_euclidean_distance


# define main function for testing
if __name__ == '__main__':

    xref = gen_circle_traj(6, np.array([0, 1]), 1)
    UOpt = np.ones(xref.shape)
    XRef_path, XSim_path, UOpt_path = store_data(xref, xref, UOpt)

    pref = '../experiment_data/example/2025-05-25 23_52_30_'
    XRef = np.load(pref+'XRef.npy')
    XSim = np.load(pref+'XSim.npy')
    UOpt = np.load(pref+'UOpt.npy')

    create_plots(1/2, XRef, XSim, UOpt, store_plots=False)
