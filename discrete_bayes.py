import copy

import numpy as np
from filterpy.discrete_bayes import normalize


def update_belief(hall, belief, z, correct_scale):
    # this is a naive implementation that uses a scaling factor instead of probablities and does not normalise
    # after update
    for i, val in enumerate(hall):
        if val == z:
            # we scale the probability of the position where the sensor matches the hallway model
            # by 3x as the sensor is three times more likely to report a correct reading
            belief[i] *= correct_scale


def scaled_update(hall, belief, z, z_prob):
    # this is a more sophisticated implementation of the update as it uses
    # vector indexing, probabilities instead of scaling and normalisation after update
    scale = z_prob / (1. - z_prob)
    posterior = copy.deepcopy(belief)
    # by multiplying the prior indices by the scale where the evidence matches the model we are
    # essentially updating the prior with the likelihood
    posterior[hall == z] *= scale
    normalize(posterior)
    return posterior


if __name__ == "__main__":
    # 1's indicate doors and 0's indicate walls
    hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    # dog is equally likely to be at any position in the hallway
    initial_prior = np.array([1 / 10] * len(hallway))

    from utils import plot_bars

    plot_bars(initial_prior)
    # we get a reading from the sensor
    reading = 1  # 1 is 'door'
    print(f"Prior before update: {initial_prior}. Sum: {sum(initial_prior)}")
    post = scaled_update(hallway, initial_prior, z=reading, z_prob=0.75)
    print(f"Prior after update and normalisation: {post}. Sum: {sum(post)}")
    plot_bars(post)
