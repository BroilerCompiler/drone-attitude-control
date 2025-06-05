import casadi as ca
from acados_template import AcadosModel
import numpy as np
from acados_template import AcadosSim, AcadosSimSolver
from gen_trajectory import gen_circle_traj, gen_static_point_traj
from store_results import create_plots
from params import ExperimentParameters, DroneData
p = ExperimentParameters()
drone_data = DroneData()


class ControllerModel:

    def __init__(self):
        '''
        Generate acados model for the crazyflie dynamics in 2D.
        Only x and z direction are considered -> only pitch angle (theta).

        u is aliased as ũ. ũ is the jerk vector from which true control inputs
        to the plant (F_d and \\theta ) are computed later.
        '''
        px = ca.SX.sym('px', 1)
        pz = ca.SX.sym('pz', 1)

        vx = ca.SX.sym('vx', 1)
        vz = ca.SX.sym('vz', 1)

        Fx = ca.SX.sym('Fx', 1)
        Fz = ca.SX.sym('Fz', 1)

        # define the dynamics
        f_expl = ca.vertcat(
            vx,
            vz,
            1/drone_data.MASS * Fx + 0,  # ax
            1/drone_data.MASS * Fz - drone_data.GRAVITY_ACC  # az
        )

        xdot = ca.SX.sym('xdot', f_expl.shape[0])

        self.model = AcadosModel()
        self.model.name = 'controllerModel'
        self.model.f_impl_expr = xdot - f_expl
        self.model.f_expl_expr = f_expl
        self.model.x = ca.vertcat(*[px, pz, vx, vz])
        self.model.xdot = xdot
        self.model.u = ca.vertcat(*[Fx, Fz])


class PlantModel:

    def __init__(self):
        '''
        Generate acados model for the crazyflie dynamics in 2D.
        Only x and z direction are considered -> only pitch angle (theta).

        (F_d and \\theta ) are controlling the system.
        '''
        px = ca.SX.sym('px', 1)
        pz = ca.SX.sym('pz', 1)

        vx = ca.SX.sym('vx', 1)
        vz = ca.SX.sym('vz', 1)

        theta = ca.SX.sym('theta', 1)
        Fd = ca.SX.sym('Fd', 1)

        # define the dynamics
        f_expl = ca.vertcat(
            vx,
            vz,
            1/drone_data.MASS * Fd * ca.sin(theta) + 0,  # ax
            1/drone_data.MASS * Fd *
            ca.cos(theta) - drone_data.GRAVITY_ACC  # az
        )

        xdot = ca.SX.sym('xdot', f_expl.shape[0])

        self.model = AcadosModel()
        self.model.name = 'plantModel'
        self.model.f_impl_expr = xdot - f_expl
        self.model.f_expl_expr = f_expl
        self.model.x = ca.vertcat(*[px, pz, vx, vz])
        self.model.xdot = xdot
        self.model.u = ca.vertcat(*[theta, Fd])


class Converter:
    def __init__(self):
        pass

    def convert(self, u_tilde):
        """Converts from the controller model (u are Forces) 
        to plant model (u are Thrust and pitch).
        Takes single u aswell as vector of u

        Args:
            u_tilde (_type_): Force vector component-wise

        Returns:
            _type_: pitch, Thrust
        """
        # input is only single u
        if len(u_tilde.shape) == 1:
            F_x, F_z = u_tilde
            theta = np.arctan2(F_x, F_z)
            F_d = np.sqrt(F_x*F_x + F_z*F_z)
            u = (theta, F_d)
        # input is a vector of u
        else:
            u = np.zeros(u_tilde.shape)
            for i in range(u_tilde.shape[0]):
                F_x = u_tilde[i, 0]
                F_z = u_tilde[i, 1]
                u[i, 0] = np.arctan2(F_x, F_z)  # theta
                u[i, 1] = np.sqrt(F_x*F_x + F_z*F_z)  # F_d
        return u


def simulate_dynamics(model, x0, u):
    '''
    Simulate the dynamics of the crazyflie

    Parameters
    ----------
    T : float
        prediction horizon
    N : int
        horizon length
    x0 : np.ndarray
        initial state
    u : np.ndarray
        control vector

    Returns
    -------
    simX : np.ndarray
        simulated ocp solution

    '''
    sim = AcadosSim()
    sim.model = model

    sim.solver_options.integrator_type = 'IRK'

    sim.solver_options.T = p.dt

    integrator = AcadosSimSolver(sim, verbose=False)

    nx = sim.model.x.shape[0]
    simX = np.ndarray((p.N+1, nx))
    simX[0, :] = x0

    for i in range(p.N):

        # Note that xdot is only used if an IRK integrator is used
        simX[i+1,
             :] = integrator.simulate(x=simX[i, :], u=u[i])

    return simX


def test_drone_dynamics(circle: bool = False):
    """Generate jerk trajectory that corresponds to a circle and
    simulate it forward (single shooting) using the model of the drone
    """

    plantModel = PlantModel()
    converter = Converter()
    nx = plantModel.model.x.shape[0]
    nu = plantModel.model.u.shape[0]

    uref_ctrl = np.zeros((p.N, nu))

    # Reference
    if circle:
        radius = 1
        omega = 2*ca.pi/p.T
        traj = gen_circle_traj(nx, center=[0, 0], radius=radius)
        for i in range(p.N):
            # acceleration ref
            uref_ctrl[i, 0] = - radius * ca.cos(omega*i/p.N*p.T)*(omega/p.T)**2
            uref_ctrl[i, 1] = - radius * \
                ca.sin(omega*i/p.N*p.T)*(omega/p.T)**2 + drone_data.GRAVITY
    else:
        static_point = [1, 0.5]
        traj = gen_static_point_traj(nx, static_point)
        uref_ctrl[:, 1] = np.ones(uref_ctrl.shape[0]) * drone_data.GRAVITY

    # start exactly on the reference
    # and simulate states over the whole trajectory using the plant model
    x0 = traj[0, :]
    uref_plant = converter.convert(uref_ctrl)
    simX = simulate_dynamics(plantModel.model, x0, uref_plant)

    # plot results
    print(f"Maximum values:")
    print(
        f"px_ref: {abs(traj[:p.N+1, 0]).max()}, px_sim: {abs(simX[:, 0]).max()}")
    print(
        f"pz_ref: {abs(traj[:p.N+1, 1]).max()}, pz_sim: {abs(simX[:, 1]).max()}")
    print(
        f"vx_ref: {abs(traj[:p.N+1, 2]).max()}, vx_sim: {abs(simX[:, 2]).max()}")
    print(
        f"vz_ref: {abs(traj[:p.N+1, 3]).max()}, vz_sim: {abs(simX[:, 3]).max()}")
    create_plots(traj, simX, uref_plant, store_plots=False)


# define main function for testing
if __name__ == '__main__':
    test_drone_dynamics(circle=True)
