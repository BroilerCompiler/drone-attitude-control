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
    Xsim = np.zeros((p.N+1, plantModel.model.x.shape[0]))
    U_opt_plant = np.zeros((p.N, plantModel.model.u.shape[0]))
    a = np.zeros((p.N, 2))

    closedLoopCost = 0
    a_i = [0, dd.GRAVITY_ACC]  # current acceleration (hover initially)
    Xsim[0] = x0

    for iteration in range(p.N):
        ocp.set_up_ocp(iteration, xref, uref)

        # Solve
        x0_bar = np.hstack((Xsim[iteration], a_i))
        ocp.ocp_solver.set(0, "lbx", x0_bar)
        ocp.ocp_solver.set(0, "ubx", x0_bar)
        status = ocp.ocp_solver.solve()
        if status != 0:
            ocp.ocp_solver.print_statistics()
            raise Exception(
                f'Failed in iteration {iteration}\nacados acados_ocp_solver returned status {status}')
        U_opt_ctrl = ocp.ocp_solver.get(0, 'u')
        X_opt = ocp.ocp_solver.get(1, 'x')
        cost = (X_opt[:4] - xref[iteration, :4]) @ np.diag([1e2, 1e2,
                                                            1e0, 1e0]) @ (X_opt[:4] - xref[iteration, :4])

        # Convert
        u_tmp, a_i = converter.convert(U_opt_ctrl, a_i)
        a[iteration] = a_i
        U_opt_plant[iteration] = u_tmp[-1]

        # Simulate next state (discard intermediate states caused by higher resolution)
        Xsim[iteration +
             1] = ocp.simulate_next_x(Xsim[iteration], u_tmp, noise)

        if verbose:
            print(
                f'{iteration}: U_opt [h_x h_z]: {np.round(U_opt_ctrl, 2)} X: {np.round(np.hstack((Xsim[iteration], a_i)), 2)} C: {np.round(cost, 5)}')

        closedLoopCost += cost

    return closedLoopCost, Xsim, a, U_opt_plant
