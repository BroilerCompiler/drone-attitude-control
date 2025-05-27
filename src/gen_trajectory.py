import numpy as np


def gen_square_traj(Nsim, nx, initial, length):
    '''
    Generate a square trajectory

    Parameters
    ----------
    Nsim : int
        number of trajectory points
    initial : np.array
        initial position
    length : float
        length of the square side

    Returns
    -------
    xref : np.array
        reference state trajectory
    '''
    xref = np.zeros((Nsim, 3))
    side = int(Nsim/4)
    xref = np.zeros((Nsim, 3))
    side = int(Nsim/4)
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


def gen_straight_traj(Nsim, T, nx, initial, length):
    '''
    Generate a straight trajectory (constant jerk)

    Parameters
    ----------
    Nsim : int
        number of trajectory points
    initial : np.array
        initial position
    length : float
        length of the trajectory

    Returns
    -------
    xref : np.array
        reference state trajectory
    '''

    # const jerk
    dt = T/Nsim
    jerk = 6*length / T**3

    xref = np.zeros((Nsim+1, nx))
    for i in range(Nsim+1):
        xref[i, 0] = initial[0] + 1/6 * jerk * (i * dt)**3
        xref[i, 1] = 0
        xref[i, 2] = 0.5 * jerk * (i * dt)**2
        xref[i, 3] = 0
        xref[i, 4] = jerk * i * dt
        xref[i, 5] = 0

    return xref


def gen_circle_traj(Nsim, T, nx, center, radius) -> np.array:
    '''
    Generate a circle trajectory

    Parameters
    ----------
    Nsim : int
        number of trajectory points
    initial : np.array
        initial position
    radius : float
        radius of the circle

    Returns
    -------
    xref : np.array
        reference state trajectory
    '''
    omega = 2*np.pi/T

    xref = np.zeros((Nsim+1, nx))
    for i in range(Nsim+1):
        # pos ref
        xref[i, 0] = center[0] + radius*np.cos(omega*i/Nsim*T)
        xref[i, 1] = center[1] + radius*np.sin(omega*i/Nsim*T)

        # velocity ref
        xref[i, 2] = -radius*np.sin(omega*i/Nsim*T)*omega/Nsim
        xref[i, 3] = radius*np.cos(omega*i/Nsim*T)*omega/Nsim

        # acceleration ref
        xref[i, 4] = - radius * np.cos(omega*i/Nsim*T)*(omega/Nsim)**2
        xref[i, 5] = - radius * np.sin(omega*i/Nsim*T)*(omega/Nsim)**2

    return xref


# define main function for testing
if __name__ == '__main__':
    N = 20
    radius = 1
    center = np.array([0, 1])
    circle = gen_circle_traj(N, 6, center, N)
    print(
        f'Coordinates of {N} sample points with radius {radius} around {center}:')
    print(circle)
