import numpy as np
from jerk_model.gen_trajectory import gen_circle_traj, gen_static_point_traj, gen_straight_traj, gen_reference_u
from store_results import create_plots
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSim, AcadosSimSolver
from plant import PlantModel
from dynamics import ControllerModel, Converter
from params import ExperimentParameters, DroneData
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
        w_x = np.array([1e2, 1e2, 1e0, 1e0, 0, 0])
        w_x_e = np.array([1e2, 1e2, 1e0, 1e0, 0, 0])
        Q = np.diag(w_x)

        w_u = np.array([1e-1]*nu)
        R = np.diag(w_u)

        self.ocp.cost.W = np.block(
            [[Q, np.zeros((nx, nu))], [np.zeros((nu, nx)), R]])
        self.ocp.cost.W_e = np.diag(w_x_e)

        # selection matrices
        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx] = np.eye(nx)
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vu[nx:] = np.eye(nu)

        self.ocp.cost.Vx_e = np.eye(nx)

        self.ocp.cost.yref = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((ny_e, ))
        """
        # set constraints
        # on u
        self.ocp.constraints.constr_type = 'BGH'
        self.ocp.constraints.constr_type_e = 'BGH'
        self.ocp.constraints.lbu = np.array(
            [dd.min_jerk, dd.min_jerk])
        self.ocp.constraints.ubu = np.array(
            [dd.max_jerk, dd.max_jerk])
        self.ocp.constraints.idxbu = np.array([0, 1])

        # set constraints
        # on x
        self.ocp.constraints.lbx = np.array(
            [dd.min_p_x, dd.min_p_z, dd.min_v_x, dd.min_v_z, dd.min_a_x, dd.min_a_z])
        self.ocp.constraints.ubx = np.array(
            [dd.max_p_x, dd.max_p_z, dd.max_v_x, dd.max_v_z, dd.max_a_x, dd.max_a_z])
        self.ocp.constraints.idxbx = np.array([0, 1, 2, 3, 4, 5])
        """

        # FIXME: do i need to pass something here?
        self.ocp.constraints.x0 = np.zeros(nx)

    def create_ocp_solver(self):

        # Solver options
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'IRK'
        self.ocp.solver_options.nlp_solver_type = 'SQP'
        self.ocp.solver_options.print_level = 0

        self.ocp.solver_options.N_horizon = p.N_horizon
        self.ocp.solver_options.tf = p.dt*p.N_horizon

        self.ocp_solver = AcadosOcpSolver(
            self.ocp, json_file=self.ocp_name+'.json', verbose=False)

    def create_simulator(self, model):

        self.sim = AcadosSim()
        self.sim.model = model
        self.sim.solver_options.T = p.dt_converter
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
    nx_plant = plantModel.model.x.shape[0]
    nx = controllerModel.model.x.shape[0]
    nu = controllerModel.model.u.shape[0]
    cps = p.ctrls_per_sample

    # gen reference
    if circle:
        radius = 1
        xref = gen_circle_traj(
            nx, center=np.array([0, 0]), radius=radius)
    else:
        length = 1
        xref = gen_straight_traj(nx, initial=[0, 0], length=length)
        xref = gen_static_point_traj(nx, initial=[0, 0.5])

    uref = gen_reference_u(nu)  # generate zeros

    # output arrays
    U_opt_ctrl = np.zeros((p.N, nu))
    U_opt_plant = np.zeros((p.N*cps, nu))
    Xsim = np.zeros((p.N*cps+1, nx_plant))
    Xsim[0] = xref[0, :nx_plant]  # [1, 0, 0, 0] later if xref[0] works
    # contains accelerations (output of converter)
    a = np.zeros((p.N, nx-nx_plant))
    a_i = [0, dd.GRAVITY_ACC]  # initial acceleration (hover)

    # create OCP
    ocp = OCP()
    ocp.create_ocp(controllerModel.model)
    ocp.create_ocp_solver()
    ocp.create_simulator(plantModel.model)

    # Solve OCP
    for iter in range(p.N):
        # Set up OCP
        for k in range(p.N_horizon):
            ocp.ocp_solver.set(k, 'yref', np.hstack(
                (xref[(iter + k)*cps], uref[iter + k])))
        ocp.ocp_solver.set(p.N_horizon, 'yref',
                           xref[(iter + p.N_horizon)*cps])

        # Solve
        x0_bar = np.hstack((Xsim[iter*cps], a[iter]))
        U_opt_ctrl[iter] = ocp.ocp_solver.solve_for_x0(x0_bar)

        # convert U_opt_ctrl to _plant
        u_tmp, a_i = converter.convert(U_opt_ctrl[iter], a_i)
        U_opt_plant[iter*cps: iter*cps + cps] = u_tmp
        a[iter] = a_i

        print(
            f'{iter}: U_opt [h_x h_z]: {np.round(U_opt_ctrl[iter], 2)} X: {np.round(Xsim[iter], 2)}')

        # Simulate next states
        # (one optimal jerk command produces multiple U_opt_plant commands)
        for i in range(p.ctrls_per_sample):
            Xsim[iter*cps+i+1, :nx_plant] = ocp.simulate_next_x(
                Xsim[iter*cps+i], U_opt_plant[iter*cps+i])

    # show results
    create_plots(xref, Xsim, U_opt_plant, store_plots=False)


# define main function for testing
if __name__ == '__main__':
    main(circle=False)
