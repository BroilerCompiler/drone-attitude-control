import numpy as np
from params import ExperimentParameters
p = ExperimentParameters()


def gen_square_traj(nx, initial, length):

    xref = np.zeros((p.N, 3))
    side = int(p.N/4)
    xref = np.zeros((p.N, 3))
    side = int(p.N/4)
    for i in range(side):
        xref[i, 0] = initial[0] + length*(i/side)
        xref[i, 1] = initial[1]
        xref[i, 2] = initial[2]
        xref[i+side, 0] = initial[0] + length
        xref[i+side, 1] = initial[1] + length*(i/side)
        xref[i+side, 2] = initial[2]
        xref[i+2*side, 0] = initial[0] + length - length*(i/side)
        xref[i+2*side, 1] = initial[1] + length
        xref[i+2*side, 2] = initial[2]
        xref[i+3*side, 0] = initial[0]
        xref[i+3*side, 1] = initial[1] + length - length*(i/side)
        xref[i+3*side, 2] = initial[2]
        xref[i, 0] = initial[0] + length*(i/side)
        xref[i, 1] = initial[1]
        xref[i, 2] = initial[2]
        xref[i+side, 0] = initial[0] + length
        xref[i+side, 1] = initial[1] + length*(i/side)
        xref[i+side, 2] = initial[2]
        xref[i+2*side, 0] = initial[0] + length - length*(i/side)
        xref[i+2*side, 1] = initial[1] + length
        xref[i+2*side, 2] = initial[2]
        xref[i+3*side, 0] = initial[0]
        xref[i+3*side, 1] = initial[1] + length - length*(i/side)
        xref[i+3*side, 2] = initial[2]

    return xref


def gen_straight_traj(nx, initial, length):
    # TODO: generate docstrings

    # const jerk
    jerk = 6*length / p.T**3

    xref = np.zeros((p.N_horizon+p.N+1, nx))
    for i in range(p.N_horizon+p.N+1):
        xref[i, 0] = initial[0] + 1/6 * jerk * (i * p.dt)**3
        xref[i, 1] = 0
        xref[i, 2] = 0.5 * jerk * (i * p.dt)**2
        xref[i, 3] = 0
        xref[i, 4] = jerk * i * p.dt
        xref[i, 5] = 0

    return xref


def gen_circle_traj(nx, center, radius) -> np.array:

    omega = 2*np.pi/p.T

    xref = np.zeros((p.N_horizon+p.N+1, nx))
    for i in range(p.N_horizon+p.N+1):
        # pos ref
        xref[i, 0] = center[0] + radius*np.cos(omega*i/p.N*p.T)
        xref[i, 1] = center[1] + radius*np.sin(omega*i/p.N*p.T)

        # velocity ref
        xref[i, 2] = -radius*np.sin(omega*i/p.N*p.T)*omega/p.N
        xref[i, 3] = radius*np.cos(omega*i/p.N*p.T)*omega/p.N

        # acceleration ref
        xref[i, 4] = - radius * np.cos(omega*i/p.N*p.T)*(omega/p.N)**2
        xref[i, 5] = - radius * np.sin(omega*i/p.N*p.T)*(omega/p.N)**2

    return xref


# define main function for testing
if __name__ == '__main__':
    radius = 1
    center = np.array([0, 1])
    circle = gen_circle_traj(6, center, radius)
    print(
        f'Coordinates of {p.N} sample points with radius {radius} around {center}:')
    print(circle)
