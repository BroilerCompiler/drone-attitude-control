import numpy as np
import casadi as ca
from acados_template import AcadosModel, AcadosSim, AcadosSimSolver
from jerk_model.gen_trajectory import gen_circle_traj, gen_straight_traj, gen_static_point_traj, gen_circle_u, gen_straight_u
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

    def convert(self, u_tilde, a_0):
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
            return self.convert_one(u_tilde, a_0)

        # input is a vector of u
        else:
            cps = p.ctrls_per_sample
            u = np.zeros(u_tilde.shape)
            u = np.concatenate([u]*cps)
            u[0:cps], a_i = self.convert_one(u_tilde[0], a_0)
            for i in range(1, u_tilde.shape[0]):
                u_tmp, a_i = self.convert_one(u_tilde[i], a_i)
                u[i*cps: i*cps + cps] = u_tmp
        return u

    def convert_one(self, u_tilde, a_i):
        h_x, h_z = u_tilde
        u = np.zeros((p.ctrls_per_sample, u_tilde.shape[0]))
        for j in range(p.ctrls_per_sample):
            a_x = (h_x * j*p.dt_converter + a_i[0])
            a_z = (h_z * j*p.dt_converter + a_i[1])
            F_x = dd.MASS * a_x
            F_z = dd.MASS * a_z
            theta = np.arctan2(F_x, F_z)
            F_d = np.sqrt(F_x*F_x + F_z*F_z)
            u[j] = [theta, F_d]

        return u, [a_x, a_z]


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

    sim.solver_options.T = p.dt_converter

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
    nx_plant = plantModel.model.x.shape[0]
    nx = controllerModel.model.x.shape[0]
    nu = controllerModel.model.u.shape[0]

    # Reference for u
    if circle:
        radius = 1
        traj = gen_circle_traj(nx_plant, center=[0, 0], radius=radius)
        uref_ctrl = gen_circle_u(nu, radius)
    else:
        length = 1
        traj = gen_straight_traj(nx_plant, initial=[0, 0], length=length)
        uref_ctrl = gen_straight_u(nu, length)

    # simulate states over the whole trajectory using the plant model
    x0 = traj[0]
    uref_plant = converter.convert(uref_ctrl, a_0=[0, +dd.GRAVITY_ACC])
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
