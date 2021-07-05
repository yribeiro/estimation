import copy
import numpy as np

from filterpy.discrete_bayes import normalize, update


def update_belief(hall, belief, z, correct_scale):
    # this is a naive implementation that uses a scaling factor instead of probablities and does not normalise
    # after update
    for i, val in enumerate(hall):
        if val == z:
            # we scale the probability of the position where the sensor matches the hallway model
            # by 3x as the sensor is three times more likely to report a correct reading
            belief[i] *= correct_scale


def _scaled_update(hall, belief, z, z_prob):
    # Note: this method is superseeded by the method below - explicitly indicating the likelihood usage

    # this is a more sophisticated implementation of the update as it uses
    # vector indexing, probabilities instead of scaling and normalisation after update
    scale = z_prob / (1. - z_prob)
    posterior = copy.deepcopy(belief)
    # by multiplying the prior indices by the scale where the evidence matches the model we are
    # essentially updating the prior with the likelihood
    posterior[hall == z] *= scale
    normalize(posterior)
    return posterior


def scaled_update(hall, belief, z, z_prob):
    # Note: this method is superseeded by the inbuilt update implementation in filterpy
    scale = z_prob / (1. - z_prob)
    lh = np.ones(len(hall))
    # the scale here is essentially the distribution over the measurement
    lh[hall == z] *= scale
    return normalize(lh * belief)


def hallway_likelihood(hall, z, z_prob):
    """ compute likelihood that a measurement matches
    positions in the hallway."""

    try:
        scale = z_prob / (1. - z_prob)
    except ZeroDivisionError:
        scale = 1e8

    lh = np.ones(len(hall))
    lh[hall == z] *= scale
    return lh


if __name__ == "__main__":
    # 1's indicate doors and 0's indicate walls
    hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    # dog is equally likely to be at any position in the hallway
    initial_prior = np.array([1 / 10] * len(hallway))

    from utils import plot_bars

    # we get a reading from the sensor
    reading = 1  # 1 is 'door'
    plot_bars(initial_prior, title=f"Prior. Sum: {sum(initial_prior):.2f}")
    likelihood = hallway_likelihood(hallway, z=reading, z_prob=0.75)
    post = update(likelihood, initial_prior)
    plot_bars(post, title=f"Posterior after update and normalisation. Sum: {sum(post):.2f}")
