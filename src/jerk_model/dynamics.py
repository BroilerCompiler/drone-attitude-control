import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
from jerk_model.gen_trajectory import gen_circle_traj, gen_straight_traj, gen_circle_u, gen_straight_u
from store_results import create_plots
from plant import PlantModel
from params import ExperimentParameters, DroneData
p = ExperimentParameters()
dd = DroneData()


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

        ax = ca.SX.sym('ax', 1)
        az = ca.SX.sym('az', 1)

        hx = ca.SX.sym('hx', 1)
        hz = ca.SX.sym('hz', 1)

        # define the dynamics
        f_expl = ca.vertcat(
            vx,
            vz,
            ax + 0,
            az - dd.GRAVITY_ACC,
            hx,
            hz
        )

        xdot = ca.SX.sym('xdot', f_expl.shape[0])

        self.model = AcadosModel()
        self.model.name = 'controllerModel_jerk'
        self.model.f_impl_expr = xdot - f_expl
        self.model.f_expl_expr = f_expl
        self.model.x = ca.vertcat(*[px, pz, vx, vz, ax, az])
        self.model.xdot = xdot
        self.model.u = ca.vertcat(*[hx, hz])


class Converter:
    def __init__(self):
        pass

    def convert(self, u_tilde):
        """Converts from the controller model (u_tilde is jerk) 
        to plant model (u are Thrust and pitch).
        Takes single u aswell as vector of u.
        Returns more controls than it took in, because the input is oversampled and 
        at every sample point the acceleration is calculated (grows linearly at constant jerk)

        Args:
            u_tilde (_type_): Force vector component-wise

        Returns:
            _type_: pitch, Thrust
        """
        # input is only single u
        if len(u_tilde.shape) == 1:
            h_x, h_z = u_tilde
            u = np.zeros((p.ctrls_per_sample, u_tilde.shape[0]))

            for j in range(p.ctrls_per_sample):
                F_x = 1/dd.MASS * (h_x * j*p.dt_converter)
                F_z = 1/dd.MASS * (h_z * j*p.dt_converter - dd.GRAVITY_ACC)
                theta = np.arctan2(F_x, F_z)
                F_d = np.sqrt(F_x*F_x + F_z*F_z)
                u[j, 0] = theta
                u[j, 1] = F_d
        # input is a vector of u
        else:
            u = np.zeros(u_tilde.shape)
            u = np.concatenate([u]*p.ctrls_per_sample)
            for i in range(u_tilde.shape[0]):
                for j in range(p.ctrls_per_sample):
                    h_x = u_tilde[i, 0]
                    h_z = u_tilde[i, 1]
                    F_x = dd.MASS * (h_x * j*p.dt_converter)
                    F_z = dd.MASS * (h_z * j*p.dt_converter - dd.GRAVITY_ACC)
                    theta = np.arctan2(F_x, F_z)
                    F_d = np.sqrt(F_x*F_x + F_z*F_z)
                    u[p.ctrls_per_sample*i + j, 0] = theta
                    u[p.ctrls_per_sample*i + j, 1] = F_d
        return u


def simulate_dynamics(model, x0, u):
    '''
    Simulate the dynamics of the crazyflie using the plant model

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
    simX = np.ndarray((p.N*p.ctrls_per_sample+1, nx))
    simX[0, :] = x0

    for i in range(p.N*p.ctrls_per_sample):
        simX[i+1, :] = integrator.simulate(x=simX[i, :], u=u[i])

    return simX


def test_drone_dynamics(circle: bool = False):
    """Generate jerk trajectory that corresponds to a circle and
    simulate it forward (single shooting) using the model of the drone
    """

    plantModel = PlantModel()
    controllerModel = ControllerModel()
    converter = Converter()
    nx = controllerModel.model.x.shape[0]
    nu = controllerModel.model.u.shape[0]

    # Reference for u
    if circle:
        radius = 1
        traj = gen_circle_traj(nx, center=[0, 0], radius=radius)
        uref_ctrl = gen_circle_u(nu, radius)
    else:
        length = 1
        traj = gen_straight_traj(nx, initial=[0, 0], length=length)
        uref_ctrl = gen_straight_u(nu, length)

    # simulate states over the whole trajectory using the plant model
    x0 = traj[0, :4]
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
    test_drone_dynamics(circle=False)
