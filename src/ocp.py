import numpy as np
import copy
from gen_trajectory import gen_circle_traj, gen_static_point_traj, compare_reftraj_vs_sim
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from dynamics import ControllerModel, PlantModel, Converter
from params import ExperimentParameters, DroneData
p = ExperimentParameters()
drone_data = DroneData()


class OCP():
    def __init__(self, ocp_name='acados_ocp'):
        self.ocp_name = ocp_name
        self.ocp = None
        self.ocp_solver = None
        self.sim = None
        self.integrator = None  # sim_solver

    def create_ocp(self, model, x0):

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

        # # set constraints
        # # on u
        # max_jerk = 0.5
        # self.ocp.constraints.constr_type = 'BGH'
        # self.ocp.constraints.constr_type_e = 'BGH'
        # self.ocp.constraints.lbu = np.array([-max_jerk, -max_jerk])
        # self.ocp.constraints.ubu = np.array([max_jerk, max_jerk])
        # self.ocp.constraints.idxbu = np.array([0, 1])

        # # set constraints
        # # on x
        # max_p = 1.5
        # max_v = 0.5
        # max_a = 0.5
        # self.ocp.constraints.lbx = np.array(
        #     [-max_p, -max_p, -max_v, -max_v, -max_a, -max_a])
        # self.ocp.constraints.ubx = np.array(
        #     [max_p, max_p, max_v, max_v, max_a, max_a])
        # self.ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5])

        self.ocp.constraints.x0 = x0

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

    def simulate_next_x(self, x0, u):

        self.integrator.set("u", u)
        self.integrator.set("x", x0)
        self.integrator.solve()

        return self.integrator.get("x")


def main(circle: bool = False):

    controllerModel = ControllerModel()
    plantModel = PlantModel()
    converter = Converter()
    nx = controllerModel.model.x.shape[0]
    nu = controllerModel.model.u.shape[0]

    # generate trajectory
    uref = np.zeros((p.N+p.N_horizon, nu))

    if circle:
        radius = 1
        xref = gen_circle_traj(nx, center=np.array([0, 0]), radius=radius)
    else:
        static_point = [1, 0.5]
        xref = gen_static_point_traj(nx, static_point)
        # hover thrust as reference
        uref[:, 1] = np.ones(uref.shape[0]) * drone_data.GRAVITY

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
    for iteration in range(p.N):
        # Set up OCP
        for k in range(p.N_horizon):
            ocp.ocp_solver.set(k, 'yref', np.hstack(
                (xref[iteration + k, :], uref[iteration + k, :])))
        ocp.ocp_solver.set(p.N_horizon, 'yref',
                           xref[iteration + p.N_horizon, :])

        U_opt_control = ocp.ocp_solver.solve_for_x0(
            x0_bar=Xsim[iteration, :])
        U_opt_plant[iteration, :] = converter.convert(U_opt_control)

        print(
            f'{iteration}: U_opt: {np.round(U_opt_plant[iteration, :], 2)} X: {np.round(Xsim[iteration, :], 2)}')

        Xsim[iteration+1, :] = ocp.simulate_next_x(
            Xsim[iteration, :], U_opt_plant[iteration, :])

    # show results
    compare_reftraj_vs_sim(t=np.linspace(0, p.T, p.N+1),
                           reftraj=xref[:p.N+1], simX=Xsim, u=U_opt_plant)


# define main function for testing
if __name__ == '__main__':
    main(circle=False)
