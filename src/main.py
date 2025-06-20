# define main function for testing
import numpy as np
import force_model.controller
import jerk_model.controller
from params import ExperimentParameters
from generate_trajectory import gen_circle_traj
from store_results import create_plots, calc_aed


def main(x0, force=True, jerk=True, noise=True, verbose=False):
    p = ExperimentParameters()

    # generate reference circle
    xref, uref = gen_circle_traj(N=p.N_conv, N_horizon=p.N_horizon *
                                 p.ctrls_per_sample, nx=6, nu=2, center=[0, 0], radius=1)

    if force:
        print("fly circle with force model")
        cost_force, xsim_force, uopt_force = force_model.controller.follow_trajectory(
            xref[:, :4], uref, x0, noise, verbose)

        res_force = [xref[:p.N_conv, :4], xsim_force[:p.N], uopt_force]
        aed_force = calc_aed(res_force[0][:, :2][::4], res_force[1][:, :2])
        print(
            f'FORCE: Total cost: {np.round(cost_force, 2)}, AvgEucDist: {aed_force}')
        create_plots(p.dt, res_force[0], res_force[1], res_force[2])
    if jerk:
        print("fly circle with jerk model")
        cost_jerk, xsim_jerk, uopt_jerk = jerk_model.controller.follow_trajectory(
            xref, uref, x0, noise, verbose)

        res_jerk = [xref[:p.N_conv, :4], xsim_jerk[:p.N_conv], uopt_jerk]
        aed_jerk = calc_aed(res_jerk[0][:, :2], res_jerk[1][:, :2])
        print(
            f'JERK: Total cost: {np.round(cost_jerk, 2)}, AvgEucDist: {aed_jerk}')
        create_plots(p.dt_conv, res_jerk[0], res_jerk[1], res_jerk[2])


if __name__ == '__main__':
    x0 = np.array([1, 0, 0, 0])
    main(x0, force=True, jerk=True, noise=True, verbose=False)
