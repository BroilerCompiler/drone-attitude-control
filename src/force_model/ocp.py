import numpy as np
import copy
from store_results import create_plots
from params import ExperimentParameters, DroneData
from plant import PlantModel
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from force_model.gen_trajectory import gen_circle_traj, gen_static_point_traj
from force_model.dynamics import ControllerModel, Converter
p = ExperimentParameters()
dd = DroneData()


class OCP():
    def __init__(self, ocp_name='acados_ocp'):
        self.ocp_name = ocp_name
        self.ocp = None
        self.ocp_solver = None
        self.sim = None
        self.integrator = None  # sim_solver

    def create_ocp(self, model):

        self.ocp = AcadosOcp()
        self.ocp.code_export_directory = 'c_generated_code_' + self.ocp_name
        self.ocp.model = model

        # set cost module
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        # get dimensions
        nx = self.ocp.model.x.size()[0]
        nu = self.ocp.model.u.size()[0]
        ny = nx + nu
        ny_e = nx

        # weighting matrices
        w_x = np.array([1e2, 1e2, 1e0, 1e0])
        w_x_e = np.array([1e2, 1e2, 1e0, 1e0])
        Q = np.diag(w_x)

        w_u = np.array([1e-1]*nu)
        R = np.diag(w_u)

        self.ocp.cost.W = np.block(
            [[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
        self.ocp.cost.W_e = np.diag(w_x_e)

        # selection matrices
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx, :] = np.eye(nx)
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vu[nx:, :] = np.eye(nu)

        self.ocp.cost.Vx_e = np.eye(nx)

        self.ocp.cost.yref = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((ny_e, ))

        # set constraints
        # on u
        self.ocp.constraints.constr_type = 'BGH'
        self.ocp.constraints.constr_type_e = 'BGH'
        self.ocp.constraints.lbu = np.array(
            [dd.min_F, dd.min_F])
        self.ocp.constraints.ubu = np.array(
            [dd.max_F, dd.max_F])
        self.ocp.constraints.idxbu = np.array([0, 1])

        # set constraints
        # on x
        self.ocp.constraints.lbx = np.array(
            [dd.min_p_x, dd.min_p_z, dd.min_v_x, dd.min_v_z])
        self.ocp.constraints.ubx = np.array(
            [dd.max_p_x, dd.max_p_z, dd.max_v_x, dd.max_v_z])
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3])

        self.ocp.constraints.x0 = np.zeros(nx)

    def create_ocp_solver(self):

        # Solver options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.nlp_solver_type = 'SQP'
        self.ocp.solver_options.print_level = 0
        # self.ocp.solver_options.qp_tol = 1e-4
        # self.ocp.solver_options.tol = 1e-5
        # self.ocp.solver_options.levenberg_marquardt = 1e-3

        self.ocp.solver_options.N_horizon = p.N_horizon
        self.ocp.solver_options.tf = p.dt*p.N_horizon

        self.ocp_solver = AcadosOcpSolver(
            self.ocp, json_file=self.ocp_name+'.json', verbose=False)

    def create_simulator(self, model):

        self.sim = AcadosSim()
        self.sim.model = model
        self.sim.solver_options.T = p.dt
        self.integrator = AcadosSimSolver(self.sim, verbose=False)

    def simulate_next_x(self, x0, u, noise):

        self.integrator.set("u", u)
        self.integrator.set("x", x0)
        self.integrator.solve()

        x_next = self.integrator.get("x")

        eps = np.random.normal(0, p.noise) if noise else 0
        return x_next + eps

    def set_up_ocp(self, iter, xref, uref):
        # Set up OCP
        for k in range(p.N_horizon):
            self.ocp_solver.set(k, 'yref', np.hstack(
                (xref[(iter + k)*p.ctrls_per_sample], uref[iter + k])))
        self.ocp_solver.set(p.N_horizon, 'yref',
                            xref[(iter + p.N_horizon)*p.ctrls_per_sample])


def test_ocp(circle: bool = False):

    controllerModel = ControllerModel()
    plantModel = PlantModel()
    converter = Converter()
    nx = controllerModel.model.x.shape[0]
    nu = controllerModel.model.u.shape[0]

    # generate trajectory
    # with hover thrust as reference
    uref = np.zeros((p.N+p.N_horizon, nu))
    uref[:, 1] = np.ones(uref.shape[0]) * dd.GRAVITY

    if circle:
        radius = 1
        xref = gen_circle_traj(nx, center=np.array([0, 0]), radius=radius)
    else:
        static_point = [1, 0.5]
        xref = gen_static_point_traj(nx, static_point)

    # output arrays
    Xsim = np.zeros((p.N+1, nx))
    Xsim[0, :] = copy.deepcopy(xref[0, :])
    U_opt_plant = np.zeros((p.N, nu))

    # create OCP
    ocp = OCP()
    ocp.create_ocp(controllerModel.model, x0=xref[0, :])
    ocp.create_ocp_solver()
    ocp.create_simulator(plantModel.model)

    # Solve OCP
    closed_loop_cost = 0
    for iteration in range(p.N):
        ocp.set_up_ocp(iteration, xref, uref)

        # Solve
        U_opt_control = ocp.ocp_solver.solve_for_x0(
            x0_bar=Xsim[iteration, :])
        closed_loop_cost += ocp.ocp_solver.get_cost()

        # convert U_opt_ctrl to _plant
        U_opt_plant[iteration, :] = converter.convert(U_opt_control)

        print(
            f'{iteration}: U_opt [theta F_d]: {np.round(U_opt_plant[iteration, :], 2)} X: {np.round(Xsim[iteration, :], 2)}')

        Xsim[iteration+1, :] = ocp.simulate_next_x(
            Xsim[iteration, :], U_opt_plant[iteration, :], noise=False)

    # # show results
    create_plots(xref, Xsim, U_opt_plant, store_plots=False, show_plots=False)
    print(f'Total COST: {closed_loop_cost}')


# define main function for testing
if __name__ == '__main__':
    test_ocp(circle=True)
