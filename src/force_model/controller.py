import numpy as np
from params import ExperimentParameters, DroneData
from plant import PlantModel
from force_model.dynamics import ControllerModel, Converter
from force_model.ocp import OCP


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
    Xsim = np.zeros((p.N+1, plantModel.model.x.shape[0]))
    U_opt_plant = np.zeros((p.N, plantModel.model.u.shape[0]))
    a = np.zeros((p.N, 2))

    closedLoopCost = 0
    Xsim[0] = x0

    for iteration in range(p.N):
        ocp.set_up_ocp(iteration, xref, uref)

        # Solve
        x0_bar = Xsim[iteration]
        ocp.ocp_solver.set(0, "lbx", x0_bar)
        ocp.ocp_solver.set(0, "ubx", x0_bar)
        status = ocp.ocp_solver.solve()
        if status != 0:
            ocp.ocp_solver.print_statistics()
            raise Exception(
                f'Failed in iteration {iteration}\nacados acados_ocp_solver returned status {status}')
        U_opt_ctrl = ocp.ocp_solver.get(0, "u")
        a[iteration] = U_opt_ctrl/dd.MASS
        X_opt = ocp.ocp_solver.get(0, 'x')
        cost = (X_opt[:4] - xref[iteration, :4]) @ np.diag([1e2, 1e2,
                                                            1e0, 1e0]) @ (X_opt[:4] - xref[iteration, :4])

        # Convert
        U_opt_plant[iteration] = converter.convert(U_opt_ctrl)

        # Simulate next state
        Xsim[iteration+1,
             :] = ocp.simulate_next_x(Xsim[iteration, :], U_opt_plant[iteration, :], noise)

        if verbose:
            print(
                f'{iteration}: U_opt [theta F_d]: {np.round(U_opt_plant[iteration, :], 2)} X: {np.round(Xsim[iteration, :], 2)} C: {cost}')

        closedLoopCost += cost

    return closedLoopCost, Xsim, a, U_opt_plant
