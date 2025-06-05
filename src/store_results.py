from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from gen_trajectory import gen_circle_traj
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


def create_plots(XRef, XSim, UOpt, store_plots=False):
    d = datetime.today().strftime('%Y-%m-%d %H_%M_%S')

    latexify_plot()

    print("Creating plots...")
    width = 3.5  # inches
    height = 3.5  # inches
    ls_ref = ':'
    ls_sim = '-'
    col_x = 'deepskyblue'
    col_x_ref = 'skyblue'
    col_u = 'darkgreen'
    col_u_ref = 'lightgreen'

    # position z over x
    plt.figure(figsize=(width, height))
    if XSim is not None:
        plt.plot(XSim[:, 0], XSim[:, 1],
                 label='$p^\\mathrm{sim}$', linestyle=ls_sim, color=col_x, alpha=0.7)
        plt.hlines(y=[dd.min_p_x, dd.max_p_x], xmin=dd.min_p_z, xmax=dd.max_p_z, linewidth=1,
                   linestyles='--', color='black')
        plt.vlines(x=[dd.min_p_z, dd.max_p_z], ymin=dd.min_p_x, ymax=dd.max_p_x, linewidth=1,
                   linestyles='--', color='black')
    if XRef is not None:
        plt.plot(XRef[:, 0], XRef[:, 1],
                 label='$p^\\mathrm{ref}$', linestyle=ls_ref, color=col_x_ref, alpha=0.7)
    if XSim is not None:
        plt.scatter(XSim[0, 0], XSim[0, 1],
                    label='Starting point', color='green')
    plt.axis('equal')
    plt.grid()
    plt.xlabel('Position (m)')
    plt.ylabel('Position (m)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if store_plots:
        plt.savefig('../experiment_data/img/' + d + '_traj_pos.pdf',
                    pad_inches=0, bbox_inches='tight')
    else:
        plt.show()

    # velocity z over x
    plt.figure(figsize=(width, height))
    if XSim is not None:
        plt.plot(XSim[:, 2], XSim[:, 3],
                 label='$v^\\mathrm{sim}$', linestyle=ls_sim, color=col_x, alpha=0.7)
        plt.hlines(y=[dd.min_v_x, dd.max_v_x], xmin=dd.min_v_z, xmax=dd.max_v_z, linewidth=1,
                   linestyles='--', color='black')
        plt.vlines(x=[dd.min_v_z, dd.max_v_z], ymin=dd.min_v_x, ymax=dd.max_v_x, linewidth=1,
                   linestyles='--', color='black')
    if XRef is not None:
        plt.plot(XRef[:, 2], XRef[:, 3],
                 label='$v^\\mathrm{ref}$', linestyle=ls_ref, color=col_x_ref, alpha=0.7)
    plt.axis('equal')
    plt.grid()
    plt.xlabel('Velocity (m/s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if store_plots:
        plt.savefig('../experiment_data/img/' + d + '_traj_vel.pdf',
                    pad_inches=0, bbox_inches='tight')
    else:
        plt.show()

    # component-wise
    width = 5  # inches
    height = 5  # inches
    xmax = UOpt.shape[0]
    x = np.arange(0, xmax*p.dt, p.dt)

    fig, ax = plt.subplots(4, sharex=True, figsize=(
        width, height), layout='constrained')

    # component-wise
    if XSim is not None:
        ax[0].plot(x, XSim[:xmax, 0],
                   label='$p^\\mathrm{sim}_\\mathrm{x}$', linestyle=ls_sim, color=col_x, alpha=0.7)
        ax[0].plot(x, XSim[:xmax, 1],
                   label='$p^\\mathrm{sim}_\\mathrm{z}$', linestyle=ls_sim, color=col_x, alpha=0.7)
        ax[1].plot(x, XSim[:xmax, 2],
                   label='$v^\\mathrm{sim}_\\mathrm{x}$', linestyle=ls_sim, color=col_x, alpha=0.7)
        ax[1].plot(x, XSim[:xmax, 3],
                   label='$v^\\mathrm{sim}_\\mathrm{z}$', linestyle=ls_sim, color=col_x, alpha=0.7)
    if XRef is not None:
        ax[0].plot(x, XRef[:xmax, 0],
                   label='$p^\\mathrm{ref}_\\mathrm{x}$', linestyle=ls_ref, color=col_x_ref, alpha=0.7)
        ax[0].plot(x, XRef[:xmax, 1],
                   label='$p^\\mathrm{ref}_\\mathrm{z}$', linestyle=ls_ref, color=col_x_ref, alpha=0.7)
        ax[1].plot(x, XRef[:xmax, 2],
                   label='$v^\\mathrm{ref}_\\mathrm{x}$', linestyle=ls_ref, color=col_x_ref, alpha=0.7)
        ax[1].plot(x, XRef[:xmax, 3],
                   label='$v^\\mathrm{ref}_\\mathrm{z}$', linestyle=ls_ref, color=col_x_ref, alpha=0.7)
    if UOpt is not None:
        ax[2].plot(x, UOpt[:xmax, 0], label='$\\theta$',
                   linestyle=ls_sim, color=col_u, alpha=0.7)
        ax[3].plot(x, UOpt[:xmax, 1],
                   label='$F_\\mathrm{d}$', linestyle=ls_sim, color=col_u, alpha=0.7)
    fig.supxlabel('Time (s)')
    fig.align_ylabels()
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    ax[0].set_ylabel('Position (m)')
    ax[1].set_ylabel('Velocity (m/s)')
    ax[2].set_ylabel('Angle (rad)')
    ax[3].set_ylabel('Thrust (N)')
    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[3].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    if store_plots:
        plt.savefig('../experiment_data/img/' + d + '_trajectory_component.pdf',
                    pad_inches=0, bbox_inches='tight')
    else:
        plt.show()

    if store_plots:
        print('Stored plots to ../experiment_data/img/')


# define main function for testing
if __name__ == '__main__':

    xref = gen_circle_traj(6, np.array([0, 1]), 1)
    UOpt = np.ones(xref.shape)
    XRef_path, XSim_path, UOpt_path = store_data(xref, xref, UOpt)

    pref = '../experiment_data/example/2025-05-25 23_52_30_'
    XRef = np.load(pref+'XRef.npy')
    XSim = np.load(pref+'XSim.npy')
    UOpt = np.load(pref+'UOpt.npy')

    create_plots(XRef, XSim, UOpt, store_plots=True)
