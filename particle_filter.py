import numpy as np
import scipy

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

        # this step is the important density from "Sequential Important Sampling" and generates the likelhood
        # in Bayes theorem
        weights *= scipy.stats.norm(distance, R).pdf(z[i])

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


if __name__ == "__main__":
    # in this script the particle state is modelled as [x, y, heading]

    # here we are not modelling the velocity because we are providing a control input
    # however, if we were passively tracking something then we would need to estimate velocity
    # and use that estimate in the predict step
    pass
