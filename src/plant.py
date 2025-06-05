import casadi as ca
from acados_template import AcadosModel
from params import DroneData
dd = DroneData()


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
            1/dd.MASS * Fd * ca.sin(theta) + 0,  # ax
            1/dd.MASS * Fd *
            ca.cos(theta) - dd.GRAVITY_ACC  # az
        )

        xdot = ca.SX.sym('xdot', f_expl.shape[0])

        self.model = AcadosModel()
        self.model.name = 'plantModel'
        self.model.f_impl_expr = xdot - f_expl
        self.model.f_expl_expr = f_expl
        self.model.x = ca.vertcat(*[px, pz, vx, vz])
        self.model.xdot = xdot
        self.model.u = ca.vertcat(*[theta, Fd])
