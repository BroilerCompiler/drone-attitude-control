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

    def convert(self, h, a_i):
        """
        Implements a_{i+1} = a_i + h_iter * dt_converter

        Converts from the controller model (h is jerk) 
        to plant model (u are Thrust and pitch).
        Takes single u aswell as vector of u.
        Returns more controls than it took in, because the input is oversampled and 
        at every sample point the acceleration is calculated (grows linearly at constant jerk)

        Args:
            h (_type_): Force vector component-wise

        Returns:
            _type_: pitch, Thrust
        """
        # input is only single u
        if len(h.shape) == 1:
            return self.convert_one(h, a_i)

        # input is a vector of u
        else:
            cps = p.ctrls_per_sample
            u = np.zeros((h.shape[0]*cps, h.shape[1]))
            a = np.zeros((h.shape))
            for i in range(h.shape[0]):
                u[i*cps: i*cps + cps], a_i = self.convert_one(h[i], a_i)
                a[i] = a_i
        return u, a

    def convert_one(self, h, a):
        u = np.zeros((p.ctrls_per_sample, 2))
        for j in range(p.ctrls_per_sample):
            a += h * p.dt_converter  # integrate over h

            F_x = dd.MASS * a[0]
            F_z = dd.MASS * a[1]
            u[j, 0] = np.arctan2(F_x, F_z)  # theta
            u[j, 1] = np.sqrt(F_x*F_x + F_z*F_z)  # F_d
        return u, a


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
    simX = np.ndarray((u.shape[0]+1, nx))
    simX[0, :] = x0

    for i in range(u.shape[0]):
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
        # uref_ctrl = np.zeros(uref_ctrl.shape) #for static point

    # simulate states over the whole trajectory using the plant model
    x0 = traj[0]
    # a is omitted here, because it is not fed back to the control model
    uref_plant, _ = converter.convert(uref_ctrl, a_i=[0, +dd.GRAVITY_ACC])
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
