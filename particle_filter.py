import numpy as np
import matplotlib.pyplot as plt

from filterpy.monte_carlo import systematic_resample
from numpy.random import uniform, randn, random
from numpy.linalg import norm
from scipy.stats import norm as scipy_norm


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


def update(particles, weights, z, R, landmarks):
    """
    Generate the likelihood weighting of the particles based on the measurements. This method calculates
    the distance of each particle from the corresponding landmarks (known) and then draws the probability
    of the measurement (z) from N(distance, R). This generates the evidence likelihood based on the current state of
    the corresponding particle.

    As a result, by evaluating N(distance_of_particle_to_landmark_i, R).pdf(z[landmark_i]) the particles that
    have state closely matching the sensor measurements will get a higher weighting.

    :param particles: Particles containing the estimated state vectors.
    :param weights: Weights of all particles.
    :param z: Measurements to all known landmarks - in the robot scenario these are distances to landmarks.
    :param R: Variance on the measurements - sensor noise.
    :param landmarks: XY positions of the landmarks (known)
    """
    for i, landmark in enumerate(landmarks):
        # for each landmark calculate the distance to all particles
        distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
        # update the corresponding weights of each particle to the landmark,
        # by the likelihood of the measurement being close to the mean

        # this step is the important density from "Sequential Important Sampling" and generates the likelihood
        # in Bayes theorem
        weights *= scipy_norm(distance, R).pdf(z[i])

    weights += 1.e-300  # avoid round-off to zero
    weights /= sum(weights)  # normalize - as specified in Bayes

    # It is important to note that we are not changing anything around the state here, we are simply
    # updating our belief in each of the particles based on the measurement

    # Based on these weights, the particles wil be resampled to generate new particles that are closer to
    # actual mean of the system


def estimate(particles, weights):
    """returns mean and variance of the weighted particles"""

    pos = particles[:, 0:2]
    # if the object we are tracking is unimodal i.e. can only be one place at any time
    # the best guess is the weighted sum of the particle parameters
    mean = np.average(pos, weights=weights, axis=0)
    var = np.average((pos - mean) ** 2, weights=weights, axis=0)
    return mean, var


def simple_resample(particles, weights):
    """
    This method does a simple resample that is based off the cummulative sum of the weights.

    np.searchsorted(a, b) uses binary search to return the first index in "a" that has a value less than "b"
    e.g. a = [0.1, 0.3, 0.4, 1.0] and b = 0.567

    The return value would be 2 for index 2 as 0.4 < 0.567.

    :param particles: Array of particles containing state vectors
    :param weights: Array of weights for each particle representing the distribution across the state
    """
    N = len(particles)
    cumulative_sum = np.cumsum(weights)
    cumulative_sum[-1] = 1.  # avoid round-off error
    # draw len(particles) number of samples from a uniform distribution between 0 and 1
    # and then use binary search to generate the indices from the cum_sum

    # the higher weighted particles occupy more of the cum_sum space and so have a higher chance of
    # being sampled
    indexes = np.searchsorted(cumulative_sum, random(N))

    # resample according to indexes - these are effectively new particles
    # NOTE: The important thing here is that the new particles contain the state of their ancestors!
    #       as a result, we have tracking across time, because these particles are then passed into the predict
    #       step
    particles[:] = particles[indexes]
    # have equal belief on all particles for the next step - this makes sense as all new particles
    # should be close to the actual state
    weights.fill(1.0 / N)


def resample_from_index(particles, weights, indexes):
    """
    Method to resample based on "indices". The method does an inplace replacement of values to save on memory
    consumption.

    :param particles: Array of particles containing state vectors
    :param weights: Array of weights for each particle representing the distribution across the state
    :param indexes: Array on indices to use to generate new particles
    :return:
    """
    particles[:] = particles[indexes]
    weights.resize(len(particles))
    weights.fill(1.0 / len(weights))


def neff(weights):
    return 1. / np.sum(np.square(weights))


def run_pf1(
        N, iters=18, sensor_std_err=.1, do_plot=True, plot_particles=False, xlim=(0, 20), ylim=(0, 20), initial_x=None
):
    """
    run the particle filter algorithm

    :param N: number of particles
    :param iters: number of simulation steps
    :param sensor_std_err: measurement std err
    :param do_plot: plot to screen
    :param plot_particles: plot all particles
    :param xlim: xlim for graph
    :param ylim: ylim for graph
    :param initial_x: optional initial mean from which particles are drawn N(initial_x, (5, 5, pi / 4))
    """
    # placeholders
    p1, p2, mu, var = None, None, None, None
    # landmarks in the world - known
    landmarks = np.array([[-1, 2], [5, 10], [12, 14], [18, 21]])
    NL = len(landmarks)

    plt.figure()

    # create particles and weights
    if initial_x is not None:
        particles = create_gaussian_particles(mean=initial_x, std=(5, 5, np.pi / 4), N=N)
    else:
        # the heading range here goes from 0 - 2*pi
        particles = create_uniform_particles(xlim, ylim, (0, 6.28), N)
    weights = np.ones(N) / N

    if plot_particles:
        alpha = .20
        if N > 5000:
            alpha *= np.sqrt(5000) / np.sqrt(N)
        plt.scatter(particles[:, 0], particles[:, 1],
                    alpha=alpha, color='g')

    xs = []
    robot_pos = np.array([0., 0.])
    for x in range(iters):
        # the actual robot moves diagonally along the XY plane in this simulation
        robot_pos += (1, 1)

        # distance from robot to each landmark
        # this step is SIMULATING a sensor measurement - in a real system this would come from
        # a radar or sonar sensor
        zs = (norm(landmarks - robot_pos, axis=1) + (randn(NL) * sensor_std_err))

        # move diagonally forward to (x+1, x+1)
        # this step is SIMULATING a contol input to the robot
        # in this case the heading change is 0 and the velocity is North East in radians/dt and m/s respectively
        predict(particles, u=(0.00, 1.414), std=(.2, .05))

        # incorporate measurements
        update(particles, weights, z=zs, R=sensor_std_err, landmarks=landmarks)

        # resample if too few effective particles - neff does this for us
        if neff(weights) < N / 2:
            indexes = systematic_resample(weights)
            resample_from_index(particles, weights, indexes)
            assert np.allclose(weights, 1 / N)
        mu, var = estimate(particles, weights)
        xs.append(mu)

        if plot_particles:
            plt.scatter(particles[:, 0], particles[:, 1], color='k', marker=',', s=1)
        p1 = plt.scatter(robot_pos[0], robot_pos[1], marker='+', color='k', s=180, lw=3)
        p2 = plt.scatter(mu[0], mu[1], marker='s', color='r')

    xs = np.array(xs)
    if do_plot:
        plt.legend([p1, p2], ['Actual', 'PF'], loc=4, numpoints=1)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        print('final position error, variance:\n\t', mu - np.array([iters, iters]), var)
        plt.show()


if __name__ == "__main__":
    # in this script the particle state is modelled as [x, y, heading]

    # here we are not modelling the velocity because we are providing a control input
    # however, if we were passively tracking something then we would need to estimate velocity
    # and use that estimate in the predict step
    run_pf1(N=10, sensor_std_err=0.5, plot_particles=True)
