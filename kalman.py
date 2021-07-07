import matplotlib.pyplot as plt
import numpy as np
from typing import NamedTuple

from utils import DogSimulation


class Gaussian(NamedTuple):
    mean: float
    var: float

    def __repr__(self):
        return f"N(mu: {self.mean:.2f}, sigma: {self.var:.2f})"


# region predict
def predict(pos: Gaussian, movement: Gaussian):
    # we can use simple addition to make the prediction
    # i.e. move the t0 -> t1 and maintain a distribution

    # the reason we can do this is because Gaussians are linear and symmetric;
    # as a result they can simply be element wise added in order to convolve the current belief
    # with the transition model

    # there is an implicit caveat here in that the time step is in 1s increments
    # as a result x + v*dt becomes x + v if dt = 1

    # if the time step changes then the velocity / movement Gaussian would be need to be scaled by "dt"

    # see the predict_move_convolution in the discrete bayes module
    return Gaussian(pos.mean + movement.mean, pos.var + movement.var)


# endregion

# region updayte
def gaussian_multiply(g1: Gaussian, g2: Gaussian):
    # the result of this operation is not a Gaussian distribution
    # but a Gaussian function

    # as a result, the gaussian represented by this mean and variance
    # will need to be normalised
    mean = (g1.var * g2.mean + g2.var * g1.mean) / (g1.var + g2.var)
    variance = (g1.var * g2.var) / (g1.var + g2.var)
    return Gaussian(mean, variance)


def update(prior: Gaussian, likelihood: Gaussian):
    # in the dog example the prior is the resulting distribution after the prediction step
    # the likelihood will be the distribution of the measurement given the state space

    posterior = gaussian_multiply(likelihood, prior)
    return posterior


# endregion

if __name__ == "__main__":
    # simulate continuous state dog tracking

    # specify transition variance and sensor variance
    transition_var = 1.  # variance in the dog's movement
    sensor_var = 1.  # variance in the sensor

    # specify initial states and transitions
    initial_state = Gaussian(0., 20. ** 2)  # dog's position, N(0, 20**2)
    velocity = 1  # 1 metre a second
    dt = 1.  # time step in seconds

    # specify models
    transition_model = Gaussian(velocity * dt, transition_var)  # displacement to add to x

    dog_sim = DogSimulation(
        x0=initial_state.mean,
        velocity=transition_model.mean,
        measurement_var=sensor_var,
        process_var=transition_model.var
    )

    # tracking variables
    true_positions = []
    predicted_positions = []
    measurement_positions = []
    estimated_positions = []

    # run simulation and tracking
    posterior = initial_state
    for _ in range(10):
        prior = predict(posterior, transition_model)
        # generate a noisy movement and a noisy sensor reading
        measurement = dog_sim.move_and_sense()
        likelihood = Gaussian(measurement, sensor_var)
        posterior = update(prior, likelihood)

        # store for plotting
        true_positions.append(dog_sim.x)
        predicted_positions.append(prior.mean)
        measurement_positions.append(measurement)
        estimated_positions.append(posterior.mean)

        # plot everything
        steps = np.arange(1, len(true_positions) + 1)

        plt.plot(steps, np.array(true_positions), "b-", label="Truth")
        plt.scatter(steps, np.array(predicted_positions), marker="*", color="r", label="Predicted")
        plt.scatter(steps, np.array(measurement_positions), marker="o", color="k", label="Measurements")
        plt.plot(steps, np.array(estimated_positions), "g-", label="Estimated")
        plt.legend()
        plt.grid()
        plt.title("Position tracking")
        plt.show(block=False)
        plt.pause(1)
        plt.cla()

    plt.plot(steps, np.array(true_positions), "b-", label="Truth")
    plt.scatter(steps, np.array(predicted_positions), marker="*", color="r", label="Predicted")
    plt.scatter(steps, np.array(measurement_positions), marker="o", color="k", label="Measurements")
    plt.plot(steps, np.array(estimated_positions), "g-", label="Estimated")
    plt.legend()
    plt.grid()
    plt.title("Position tracking")
    plt.show()
