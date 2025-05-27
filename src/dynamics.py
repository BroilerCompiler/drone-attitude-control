import casadi as ca
from acados_template import AcadosModel
import numpy as np
from matplotlib import pyplot as plt
from acados_template import AcadosSim, AcadosSimSolver
from gen_trajectory import gen_circle_traj, gen_straight_traj
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
    sim.solver_options.num_stages = 4
    sim.solver_options.num_steps = 3
    sim.solver_options.newton_iter = 8  # for implicit integrator
    sim.solver_options.collocation_type = "GAUSS_RADAU_IIA"

    sim.solver_options.T = p.T

    integrator = AcadosSimSolver(sim, verbose=False)

    nx = sim.model.x.shape[0]
    simX = np.ndarray((p.N+1, nx))
    simX[0, :] = x0

    for i in range(p.N):

        # Note that xdot is only used if an IRK integrator is used

        simX[i+1,
             :] = integrator.simulate(x=simX[i, :], u=u[i], xdot=np.concatenate([simX[i, 2:], u[i]]))

    return simX


def compare_reftraj_vs_sim(t, reftraj, simX, u):
    _, ax0 = plt.subplots(2, 2)
    fig1, ax1 = plt.subplots(4, sharex=True)
    # z over x
    ax0[0, 0].plot(simX[:, 0], simX[:, 1], label='p_sim')
    ax0[0, 1].plot(simX[:, 2], simX[:, 3], label='v_sim')
    ax0[1, 0].plot(simX[:, 4], simX[:, 5], label='a_sim')
    ax0[0, 0].plot(reftraj[:, 0], reftraj[:, 1], label='p_ref')
    ax0[0, 1].plot(reftraj[:, 2], reftraj[:, 3], label='v_ref')
    ax0[1, 0].plot(reftraj[:, 4], reftraj[:, 5], label='a_ref')
    ax0[1, 1].plot(u[:, 0], u[:, 1], label='h')

    # component-wise
    ax1[0].plot(t, simX[:, 0], label='p_sim_x')
    ax1[1].plot(t, simX[:, 2], label='v_sim_x')
    ax1[2].plot(t, simX[:, 4], label='a_sim_x')
    ax1[0].plot(t, simX[:, 1], label='p_sim_z')
    ax1[1].plot(t, simX[:, 3], label='v_sim_z')
    ax1[2].plot(t, simX[:, 5], label='a_sim_z')
    ax1[0].plot(t, reftraj[:, 0], label='p_ref_x')
    ax1[1].plot(t, reftraj[:, 2], label='v_ref_x')
    ax1[2].plot(t, reftraj[:, 4], label='a_ref_x')
    ax1[0].plot(t, reftraj[:, 1], label='p_ref_z')
    ax1[1].plot(t, reftraj[:, 3], label='v_ref_z')
    ax1[2].plot(t, reftraj[:, 5], label='a_ref_z')
    ax1[3].plot(t[:-1], u[:, 0], label='h_x')
    ax1[3].plot(t[:-1], u[:, 1], label='h_z')
    fig1.supxlabel('Seconds')
    ax0[0, 0].legend()
    ax0[0, 1].legend()
    ax0[1, 0].legend()
    ax0[1, 1].legend()
    ax1[0].legend()
    ax1[1].legend()
    ax1[2].legend()
    ax1[3].legend()
    ax0[0, 0].grid()
    ax0[0, 1].grid()
    ax0[1, 0].grid()
    ax0[1, 1].grid()
    ax1[0].grid()
    ax1[1].grid()
    ax1[2].grid()
    ax1[3].grid()
    ax0[0, 0].axis('equal')
    ax0[0, 1].axis('equal')
    ax0[1, 0].axis('equal')
    ax0[1, 1].axis('equal')

    plt.show()


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
        omega = 2*np.pi/p.T
        traj = gen_circle_traj(nx, center=[0, 0], radius=radius)
        for i in range(p.N):
            u_vec[i, 0] = radius * np.sin(omega*i/p.N*p.T)*(omega/p.N)**3
            u_vec[i, 1] = - radius * np.cos(omega*i/p.N*p.T)*(omega/p.N)**3
    else:
        length = 1
        traj = gen_straight_traj(nx, initial=[0, 0], length=length)
        jerk = 6*length / p.T**3
        u_vec[:, 0] = np.ones(p.N) * jerk

    # start exactly on the reference
    x0 = traj[0, :]
    simX = simulate_dynamics(drone.model, x0, u_vec)
    compare_reftraj_vs_sim(t=np.linspace(0, p.T, p.N+1),
                           reftraj=traj[:p.N+1], simX=simX, u=u_vec)


# define main function for testing
if __name__ == '__main__':
    test_drone_dynamics(circle=False)
