from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from gen_trajectory import gen_circle_traj
from acados_template.plot_utils import latexify_plot
from params import ExperimentParameters
p = ExperimentParameters()


def store_data(XRef, XSim, UOpt):
    d = datetime.today().strftime('%Y-%m-%d %H_%M_%S')
    XRef_path, XSim_path, UOpt_path = f'../experiment_data/npy/{d}_XRef.npy', f'../experiment_data/npy/{d}_XSim.npy', f'../experiment_data/npy/{d}_UOpt.npy'
    np.save(XRef_path, XRef)
    np.save(XSim_path, XSim)
    np.save(UOpt_path, UOpt)
    # np.save(f'sol_conv{d}.npy', solver_converged)
    print(f'Stored data to ../experiment_data/npy/')
    return XRef_path, XSim_path, UOpt_path


def create_plots(XRef_path, XSim_path, UOpt_path, store_plots=False):
    XRef = np.load(XRef_path) if XRef_path else None
    XSim = np.load(XSim_path) if XSim_path else None
    UOpt = np.load(UOpt_path) if UOpt_path else None
    d = datetime.today().strftime('%Y-%m-%d %H_%M_%S')

    latexify_plot()

    print("Creating plots...")

    # plot traj vs reference
    width = 3.5  # inches
    height = 3.5  # inches
    plt.figure(figsize=(width, height))
    plt.plot(XSim[:, 0], XSim[:, 1], label=r'$p$')
    plt.plot(XRef[:, 0], XRef[:, 1], label=r'$p^\mathrm{ref}$')
    plt.scatter(XRef[0, 0], XRef[0, 1], label=r'Starting point', color='green')
    plt.legend()
    plt.grid()
    plt.xlabel(r'$p_\mathrm{x}$ [m]')
    plt.ylabel(r'$p_\mathrm{z}$ [m]')
    ax = plt.gca()
    ax.set_aspect('equal')
    if store_plots:
        plt.savefig('../experiment_data/img/' + d + '_trajectory.pdf',
                    pad_inches=0, bbox_inches='tight')
    else:
        plt.show()

    # plot traj vs reference component-wise
    width = 3.5  # inches
    height = 5  # inches
    xmax = UOpt.shape[0]
    x = np.arange(0, xmax*p.dt, p.dt)
    fig, ax = plt.subplots(4, sharex='col', figsize=(width, height))
    ax[0].plot(x, XSim[:xmax, 0], label=r'$p_\mathrm{x}$')
    ax[0].plot(x, XRef[:xmax, 0], label=r'$p_\mathrm{x}^\mathrm{ref}$')
    ax[1].plot(x, XSim[:xmax, 1], label=r'$p_\mathrm{z}$')
    ax[1].plot(x, XRef[:xmax, 1], label=r'$p_\mathrm{z}^\mathrm{ref}$')
    ax[2].plot(x, UOpt[:xmax, 0], label=r'$h_\mathrm{x}$')
    ax[3].plot(x, UOpt[:xmax, 1], label=r'$h_\mathrm{z}$')
    ax[0].set_xlim([0, xmax*p.dt])
    ax[1].set_xlim([0, xmax*p.dt])
    ax[2].set_xlim([0, xmax*p.dt])
    ax[3].set_xlim([0, xmax*p.dt])
    ax[0].set_ylabel(r'$p_\mathrm{x}$ [m]')
    ax[1].set_ylabel(r'$p_\mathrm{z}$ [m]')
    ax[2].set_ylabel(r'$h_\mathrm{z}$ [m/s3]')
    ax[3].set_ylabel(r'$h_\mathrm{z}$ [m/s3]')
    ax[0].legend()
    ax[1].legend()
    ax[2].legend()
    ax[3].legend()
    ax[0].grid()
    ax[1].grid()
    ax[2].grid()
    ax[3].grid()
    fig.supxlabel('Seconds')
    fig.align_ylabels()
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
    create_plots(pref+'XRef.npy', pref+'XSim.npy',
                 pref+'UOpt.npy', store_plots=True)
