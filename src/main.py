# define main function for testing
import numpy as np
import force_model.controller
import jerk_model.controller
from params import ExperimentParameters
from generate_trajectory import gen_circle_traj
from store_results import create_plots, calc_aed


def main(x0, force=True, jerk=True, noise=True, plots=True, verbose=False):
    p = ExperimentParameters()

    # generate reference circle
    ref = gen_circle_traj(p.N, p.N_horizon, nx=6, nu=2,
                          center=[0, 0], radius=1)

    if force:
        print("fly circle with force model")
        cost_force, xsim_force, a_force, uopt_force = force_model.controller.follow_trajectory(
            ref[:, :4], ref[:, 4:6], x0, noise, verbose)

        res_force = [ref[:p.N], xsim_force[:p.N], a_force, uopt_force]
        aed_force = calc_aed(res_force[0][:, :2], res_force[1][:, :2])
        print(
            f'FORCE: Total cost: {np.round(cost_force, 2)}, AvgEucDist: {aed_force}')
        if plots:
            create_plots(
                p.dt, res_force[0], res_force[1], None, res_force[3])
    if jerk:
        print("fly circle with jerk model")
        cost_jerk, xsim_jerk, a_jerk, uopt_jerk = jerk_model.controller.follow_trajectory(
            ref[:, :6], ref[:, 6:], x0, noise, verbose)

        res_jerk = [ref[:p.N], xsim_jerk[:p.N], a_jerk, uopt_jerk]
        aed_jerk = calc_aed(res_jerk[0][:, :2], res_jerk[1][:, :2])
        print(
            f'JERK: Total cost: {np.round(cost_jerk, 2)}, AvgEucDist: {aed_jerk}')
        if plots:
            create_plots(p.dt, res_jerk[0],
                         res_jerk[1], res_jerk[2], res_jerk[3])


if __name__ == '__main__':
    np.random.seed(42)
    x0 = np.array([1.0, 0, 0, 0.62])
    main(x0, force=True, jerk=True, noise=True, plots=True, verbose=False)
