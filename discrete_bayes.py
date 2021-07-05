import copy
import time

import numpy as np
import matplotlib.pyplot as plt

from filterpy.discrete_bayes import normalize, update, predict

from utils import plot_bars


# region update methods - sensor models and uncertainty

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


# endregion

# region predict methods - i.e. transition models and uncertainty

def perfect_predict(belief, move):
    """ move the position by `move` spaces, where positive is
    to the right, and negative is to the left
    """
    # this method assumes that the move (transition / dynamics model) has zero noise
    # or the sensor reporting the move action has 0 noise

    # as a result we simply shift values to the right (circular at the end)
    n = len(belief)
    result = np.zeros(n)
    for i in range(n):
        # this indexing structure is pretty genius
        # the module returns the remainder after integer division by n
        # the formula is x - n * floor(x / n)

        # from above negative numbers i.e. -1 % 10 = 9
        # as a result the 9th position is placed in the 0th position of the new array

        # one worth remembering
        result[i] = belief[(i - move) % n]
    return result


def predict_move(belief, move, p_under, p_correct, p_over):
    # in this function we are convolving the belief with some kernel i.e. [p_under, p_correct, p_over]
    # to characterise a movement uncertainty distribution

    # note: if we keep convolving with the same kernel over and over again the distribution
    # eventually degrades to uniform until all information is lost
    n = len(belief)
    prior = np.zeros(n)
    for i in range(n):
        prior[i] = (
                belief[(i - move) % n] * p_correct +
                belief[(i - move - 1) % n] * p_over +
                belief[(i - move + 1) % n] * p_under)
    return prior


def predict_move_convolution(pdf, offset, kernel):
    # Note: this method implements convolution of a general kernel on a pdf function
    # this method is superseeded by the predict method in filterpy
    N = len(pdf)
    kN = len(kernel)
    width = int((kN - 1) / 2)

    prior = np.zeros(N)
    for i in range(N):
        for k in range(kN):
            index = (i + (width - k) - offset) % N
            # this line here is essentially convolving the current best guess of the state
            # with the uncertainty of the prediction (kernel) to get to the next state
            prior[i] += pdf[index] * kernel[k]
    # the variable is named prior here, as the predict step with the transition
    # creates a new prior to be updated with the likelihood and evidence

    # the transition is as follows:
    # > prior_t0 * likelihood_t0 -> posterior_t0 (this is the estimate incorporating evidence)
    # > posterior_t0 * transition_model -> prior_t1 (this is the predict step)
    # > prior_t1 * likelihood_t1 -> posterior_t1 (this is the update step as before)
    return prior


# endregion

if __name__ == "__main__":
    # 1's indicate doors and 0's indicate walls
    hallway = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    # # dog is equally likely to be at any position in the hallway
    # initial_prior = np.array([1 / 10] * len(hallway))
    #
    # # we get a reading from the sensor
    # reading = 1  # 1 is 'door'
    # plot_bars(initial_prior, title=f"Prior. Sum: {sum(initial_prior):.2f}")
    # likelihood = hallway_likelihood(hallway, z=reading, z_prob=0.75)
    # post = update(likelihood, initial_prior)
    # plot_bars(post, title=f"Posterior after update and normalisation. Sum: {sum(post):.2f}")

    # # moving belief's based on transitions
    # belief = np.array([.35, .1, .2, .3, 0, 0, 0, 0, 0, .05])
    #
    # plt.subplot(121)
    # plot_bars(belief, title='Before prediction', ylim=(0, .4), show=False)
    # move = 1
    # belief = perfect_predict(belief, move=move)
    # plt.subplot(122)
    # plot_bars(belief, title=f'After prediction. move={move}', ylim=(0, .4))

    # moving belief based on uncertain transitions
    initial_prior = np.array([1 / len(hallway)] * len(hallway))
    move = 1  # move 1 to the right
    starting_index = 0
    true_position = np.array([0.0] * len(hallway))
    true_position[starting_index] = 1.0

    # simulate motion moving through hallway - the filter eventually settles on picking the position
    for i in range(100):
        # get the likelihood based on the measurement (75% chance of correct)
        likelihood = hallway_likelihood(hallway, z=hallway[starting_index], z_prob=0.8)
        posterior = update(likelihood, initial_prior)
        plot_bars(initial_prior, title=f"Posterior estimates. Moving @ {move} per time step", show=True, ylim=(0, 1.2))
        plt.scatter(np.array([i for i in range(len(hallway))]), true_position, color="r")
        plt.xticks(
            np.array([i for i in range(len(hallway))]),
            np.array(["Door" if a == 1 else "Wall" for a in hallway])
        )
        plt.pause(1)
        plt.cla()

        # convert the posterior into a prior with a motion model with a kernel uncertainty
        prior_after_transition = predict(posterior, move, [0.05, 0.05, 0.8, 0.05, 0.05])
        # update the prior
        initial_prior = prior_after_transition

        # move every time step - i.e. send control command to the machine
        # move the index to the right
        starting_index += move
        starting_index %= len(hallway)
        true_position = np.array([0.0] * len(hallway))
        true_position[starting_index] = 1.0
