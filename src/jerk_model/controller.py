import numpy as np
from params import ExperimentParameters, DroneData
from plant import PlantModel
from jerk_model.dynamics import ControllerModel, Converter
from jerk_model.ocp import OCP


def follow_trajectory(xref, uref, x0, noise, verbose=True):
    p = ExperimentParameters()
    dd = DroneData()
    plantModel = PlantModel(noise)
    controllerModel = ControllerModel()
    converter = Converter()
    ocp = OCP()
    ocp.create_ocp(controllerModel.model)
    ocp.create_ocp_solver()
    ocp.create_simulator(plantModel.model)
    Xsim = np.zeros((p.N_conv+1, plantModel.model.x.shape[0]))
    U_opt_plant = np.zeros((p.N_conv, plantModel.model.u.shape[0]))

    closedLoopCost = 0
    a_i = [0, dd.GRAVITY_ACC]  # current acceleration (hover initially)
    Xsim[0] = x0

    for iteration in range(p.N):
        cps = p.ctrls_per_sample
        ocp.set_up_ocp(iteration, xref, uref)

        # Solve
        x0_bar = np.hstack((Xsim[iteration*cps], a_i))
        U_opt_ctrl = ocp.ocp_solver.solve_for_x0(x0_bar)
        # ocp.ocp_solver.set(0, "lbx", x0_bar)
        # ocp.ocp_solver.set(0, "ubx", x0_bar)
        # status = ocp.ocp_solver.solve()
        # if status != 0:
        #     ocp.ocp_solver.print_statistics()
        #     raise Exception(
        #         f'Failed in iteration {iteration}\nacados acados_ocp_solver returned status {status}')
        # U_opt_ctrl = ocp.ocp_solver.get(0, "u")
        cost = ocp.ocp_solver.get_cost()

        # Convert
        u_tmp, a_i = converter.convert(U_opt_ctrl, a_i)
        U_opt_plant[iteration*cps: iteration*cps + cps] = u_tmp

        # Simulate next states
        # (one optimal jerk command produces multiple U_opt_plant commands and therefore more )

        for i in range(cps):
            Xsim[iteration*cps+i +
                 1] = ocp.simulate_next_x(Xsim[iteration*cps+i], U_opt_plant[iteration*cps+i], noise)

        if verbose:
            print(
                f'{iteration}: U_opt [h_x h_z]: {np.round(U_opt_ctrl, 2)} X: {np.round(np.hstack((Xsim[iteration*cps], a_i)), 2)} C: {cost}')

        closedLoopCost += cost

    return closedLoopCost, Xsim, U_opt_plant
