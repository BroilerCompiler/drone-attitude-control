import numpy as np
from params import ExperimentParameters
p = ExperimentParameters()


def gen_circle_traj(N, N_horizon, nx, nu, center, radius):
    xref = np.ndarray((N+N_horizon, nx))

    omega = 2*np.pi/p.T
    i = np.linspace(0, p.T, p.N_conv)

    xref[:N, 0] = center[0] + radius * np.cos(omega*i)
    xref[:N, 1] = center[1] + radius * np.sin(omega*i)
    if nx == 4 or nx == 6:
        xref[:N, 2] = -radius * omega * np.sin(omega*i)
        xref[:N, 3] = radius * omega * np.cos(omega*i)
    elif nx == 6:
        xref[:N, 4] = -radius * omega**2 * np.cos(omega*i)
        xref[:N, 5] = -radius * omega**2 * np.sin(omega*i)
    else:
        raise Exception('Invalid dimensions')

    xref[N:] = xref[:N_horizon]

    uref = np.zeros((N+N_horizon, nu))

    return xref, uref
