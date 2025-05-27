import casadi as ca
from acados_template import AcadosModel
import numpy as np
from acados_template import AcadosSim, AcadosSimSolver
from gen_trajectory import gen_circle_traj, gen_straight_traj, compare_reftraj_vs_sim
from params import ExperimentParameters
p = ExperimentParameters()


class DroneDynamics:

    def __init__(self):
        '''
        Generate acados model for the crazyflie dynamics in 2D.
        Only x and z direction are considered -> only pitch angle (theta).

        u is aliased as ũ. ũ is the jerk vector from which true control inputs
        to the plant (F_d and \\theta ) are computed later.
        '''
        GRAVITY_WORLD_COORD = -9.81

        px = ca.SX.sym('px', 1)
        pz = ca.SX.sym('pz', 1)

        vx = ca.SX.sym('vx', 1)
        vz = ca.SX.sym('vz', 1)

        a_omega_x = ca.SX.sym('a_omega_x', 1)
        a_omega_z = ca.SX.sym('a_omega_z', 1)

        hx = ca.SX.sym('hx', 1)
        hz = ca.SX.sym('hz', 1)

        ax = a_omega_x  # + p.dt*hx
        az = a_omega_z  # + p.dt*hz + GRAVITY_WORLD_COORD

        # define the dynamics
        f_expl = ca.vertcat(
            vx,
            vz,
            ax,
            az,
            hx,
            hz
        )

        xdot = ca.SX.sym('xdot', f_expl.shape[0])

        self.model = AcadosModel()
        self.model.name = 'drone_pointmass_model'
        self.model.f_impl_expr = xdot - f_expl
        self.model.f_expl_expr = f_expl
        self.model.x = ca.vertcat(*[px, pz, vx, vz, ax, az])
        self.model.xdot = xdot
        self.model.u = ca.vertcat(*[hx, hz])


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
             :] = integrator.simulate(x=simX[i, :], u=u[i], xdot=np.concatenate([simX[i, 2:], u[i]]))

    return simX


def test_drone_dynamics(circle: bool = False):
    """Generate jerk trajectory that corresponds to a circle and
    simulate it forward (single shooting) using the model of the drone
    """

    drone = DroneDynamics()
    nx = drone.model.x.shape[0]
    nu = drone.model.u.shape[0]

    u_vec = np.zeros((p.N, nu))

    # Reference
    if circle:
        radius = 1
        omega = 2*ca.pi/p.T
        traj = gen_circle_traj(nx, center=[0, 0], radius=radius)
        for i in range(p.N):
            u_vec[i, 0] = radius * ca.sin(omega*i/p.N*p.T)*(omega)**3
            u_vec[i, 1] = - radius * ca.cos(omega*i/p.N*p.T)*(omega)**3
    else:
        length = 1
        traj = gen_straight_traj(nx, initial=[0, 0], length=length)
        jerk = 6*length / p.T**3
        u_vec[:, 0] = np.ones(p.N) * jerk

    # start exactly on the reference
    x0 = traj[0, :]
    simX = simulate_dynamics(drone.model, x0, u_vec)

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
    print(
        f"ax_ref: {abs(traj[:p.N+1, 4]).max()}, ax_sim: {abs(simX[:, 4]).max()}")
    print(
        f"az_ref: {abs(traj[:p.N+1, 5]).max()}, az_sim: {abs(simX[:, 5]).max()}")
    compare_reftraj_vs_sim(t=np.linspace(0, p.T, p.N+1),
                           reftraj=traj[:p.N+1], simX=simX, u=u_vec)


# define main function for testing
if __name__ == '__main__':
    test_drone_dynamics(circle=True)
