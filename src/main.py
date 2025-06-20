# define main function for testing
import numpy as np
import force_model.controller
import jerk_model.controller
from params import ExperimentParameters
from generate_trajectory import gen_circle_traj
from store_results import create_plots


def main(force=True, jerk=True, verbose=False):
    p = ExperimentParameters()

    # generate reference circle
    xref, uref = gen_circle_traj(N=p.N_conv, N_horizon=p.N_horizon *
                                 p.ctrls_per_sample, nx=6, nu=2, center=[0, 0], radius=1)
    x0 = xref[0, :4]

    if force:
        print("fly circle with force model")
        cost_force, xsim_force, uopt_force = force_model.controller.follow_trajectory(
            xref[:, :4], uref, x0, verbose)
        print(f'FORCE: {np.round(cost_force, 2)}')
        create_plots(p.dt, xref[:p.N_conv, :4], xsim_force[:p.N], uopt_force)
    if jerk:
        print("fly circle with jerk model")
        cost_jerk, xsim_jerk, uopt_jerk = jerk_model.controller.follow_trajectory(
            xref, uref, x0, verbose)

        print(f'JERK: {np.round(cost_jerk, 2)}')
        create_plots(p.dt_conv, xref[:p.N_conv, :4],
                     xsim_jerk[:p.N_conv], uopt_jerk)


if __name__ == '__main__':
    main(force=True, jerk=True, verbose=False)
