import numpy as np

from numpy.random import uniform, randn


def create_uniform_particles(x_range, y_range, hdg_range, N):
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], size=N)
    particles[:, 1] = uniform(y_range[0], y_range[1], size=N)
    particles[:, 2] = uniform(hdg_range[0], hdg_range[1], size=N)
    particles[:, 2] %= 2 * np.pi
    return particles


def create_gaussian_particles(mean, std, N):
    particles = np.empty((N, 3))
    # multiplying a 0 mean normal distribution and multiplying by std deviation
    # and adding to the mean if the same as drawing from the parametrised function
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


def predict(particles, u, std, dt=1.):
    """ move according to control input u (heading change, velocity)
    with noise Q (std heading change, std velocity) """

    # NOTE: here we are generating a random control signal based around the mean heading change and velocity
    #       the reason is that there is noise in the robot control and process, and so the control signal
    #       will not result in the exact end state desired
    #
    #       this is important as the particle filter will fail to estimate the joint distribution if the
    #       process model does not have noise on it

    N = len(particles)
    # update heading - add the heading change and then normalise to 0 -> 360
    particles[:, 2] += u[0] + (randn(N) * std[0])
    particles[:, 2] %= 2 * np.pi

    # move in the (noisy) commanded direction - calculate distance move vector and split into X and Y components
    dist = (u[1] * dt) + (randn(N) * std[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist

if __name__ == "__main__":
    # in this script the particle state is modelled as [x, y, heading]

    # here we are not modelling the velocity because we are providing a control input
    # however, if we were passively tracking something then we would need to estimate velocity
    # and use that estimate in the predict step
    pass