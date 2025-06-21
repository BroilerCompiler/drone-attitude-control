'''
loads constants of the drone, like inertia and mass
'''

import os
import numpy as np
import xml.etree.ElementTree as etxml


class DroneData:
    URDF_DIR = os.path.dirname(os.path.abspath(__file__))

    def __init__(self, urdf_filename: str = 'cf2x.urdf'):
        self.URDF_PATH = os.path.join(self.URDF_DIR, urdf_filename)

        self.MASS, \
            self.L, \
            self.THRUST2WEIGHT_RATIO, \
            self.J, \
            self.J_INV, \
            self.KF, \
            self.KM, \
            self.COLLISION_H, \
            self.COLLISION_R, \
            self.COLLISION_Z_OFFSET, \
            self.MAX_SPEED_KMH, \
            self.GND_EFF_COEFF, \
            self.PROP_RADIUS, \
            self.DRAG_COEFF, \
            self.DW_COEFF_1, \
            self.DW_COEFF_2, \
            self.DW_COEFF_3, \
            self.PWM2RPM_SCALE, \
            self.PWM2RPM_CONST, \
            self.MIN_PWM, \
            self.MAX_PWM = self._parse_urdf_parameters(self.URDF_PATH)
        self.GRAVITY_ACC = 9.81
        self.RAD2DEG = 180 / np.pi
        self.DEG2RAD = np.pi / 180
        self.L_TETHER = 0.435  # [m]
        self.H_POLE = 1.16  # [m]
        self.MASS = 0.03277  # drone mass with lighthouse and tether
        self.X_LOADCELL = 0.14  # [m]
        self.BAUMGARTE_KAPPA = 1.0
        self.GRAVITY = self.GRAVITY_ACC * self.MASS
        self.max_F = 1.3 * self.GRAVITY
        self.min_F = -0.2 * self.GRAVITY
        self.min_p_x = -2
        self.max_p_x = 2
        self.min_p_z = -2
        self.max_p_z = 2
        self.min_v_x = -1
        self.max_v_x = 1
        self.min_v_z = -1
        self.max_v_z = 1
        self.min_a_x = -15
        self.max_a_x = 15
        self.min_a_z = -15
        self.max_a_z = 15
        self.min_jerk = -5
        self.max_jerk = 5
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4 * self.KF))
        self.MAX_RPM = np.sqrt(
            (self.THRUST2WEIGHT_RATIO * self.GRAVITY) / (4 * self.KF))
        self.MAX_THRUST = (4 * self.KF * self.MAX_RPM**2)
        self.HOVER_THRUST = (4 * self.KF * self.HOVER_RPM**2)
        # Assuming that it's half of hover thrust
        self.MIN_THRUST = (4 * self.KF * self.HOVER_RPM**2) / 2
        self.MAX_XY_TORQUE = (self.L * self.KF * self.MAX_RPM**2)
        self.MAX_Z_TORQUE = (2 * self.KM * self.MAX_RPM**2)

    def _parse_urdf_parameters(self, file_name):
        '''Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.
        '''
        URDF_TREE = etxml.parse(file_name).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [
            float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')
        ]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        PWM2RPM_SCALE = float(URDF_TREE[0].attrib['pwm2rpm_scale'])
        PWM2RPM_CONST = float(URDF_TREE[0].attrib['pwm2rpm_const'])
        MIN_PWM = float(URDF_TREE[0].attrib['pwm_min'])
        MAX_PWM = float(URDF_TREE[0].attrib['pwm_max'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
            GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3, \
            PWM2RPM_SCALE, PWM2RPM_CONST, MIN_PWM, MAX_PWM


class ExperimentParameters:
    def __init__(self):
        self.T = 10
        self.dt = 1/50  # dt of the MPC
        self.dt_conv = 1/200  # lower level can be faster -> better performance
        self.ctrls_per_sample = int(self.dt / self.dt_conv)
        self.N = int(self.T/self.dt)
        self.N_conv = int(self.T/self.dt_conv)
        self.N_horizon = 30
        self.noise = 0.005


# define main function for testing
if __name__ == '__main__':
    drone_data = DroneData()
    print(f'Drone mass is {drone_data.MASS}')
