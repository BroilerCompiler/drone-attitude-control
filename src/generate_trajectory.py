import numpy as np
from params import ExperimentParameters, DroneData
p = ExperimentParameters()
dd = DroneData()


def gen_circle_traj(N, N_horizon, nx, nu, center, radius):
    ref = np.ndarray((N+N_horizon, nx+nu), dtype=float)

    omega = 2*np.pi/p.T
    i = np.linspace(0, p.T, N)

    ref[:N, 0] = center[0] + radius * np.cos(omega*i)
    ref[:N, 1] = center[1] + radius * np.sin(omega*i)
    ref[:N, 2] = -radius * omega * np.sin(omega*i)
    ref[:N, 3] = radius * omega * np.cos(omega*i)
    if nx == 4 or nx == 6:
        ref[:N, 4] = -radius * omega**2 * np.cos(omega*i)
        ref[:N, 5] = -radius * omega**2 * np.sin(omega*i) + dd.GRAVITY_ACC
    else:
        raise Exception('Invalid dimensions')
    if nx == 6:
        ref[:N, 6] = 0
        ref[:N, 7] = 0

    ref[N:] = ref[:N_horizon]

    return ref
