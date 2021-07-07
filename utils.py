from math import sqrt
from typing import Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def plot_data(estimates, predictions, measurements, truth):
    # all data is from t1 onwards .. except truth
    estimates = np.array(estimates)
    measurements = np.array(measurements)
    predictions = np.array(predictions)
    truth = np.array(truth)
    time = np.array(range(1, len(measurements) + 1))

    plt.plot(truth, ls="-", color="k", label="Truth")
    plt.plot(time, estimates, ls=":", color="b", label="Estimates")
    plt.scatter(time, measurements, marker="o", color="k", label="Measurements")
    plt.scatter(time, predictions, marker="^", color="r", label="Predictions")
    plt.grid()
    plt.legend()
    plt.show()


def plot_bars(arr: np.ndarray, title: Optional[str] = None, ylim: Optional[Tuple] = None, show=True):
    x = np.array([i for i in range(len(arr))])
    if title:
        plt.title(label=title)
    plt.bar(x, arr)
    if ylim:
        plt.ylim(*ylim)
    else:
        plt.ylim(0, 1)
    plt.grid()
    if show:
        plt.show(block=False)


class DogSimulation(object):
    def __init__(self, x0=0, velocity=1,
                 measurement_var=0.0,
                 process_var=0.0):
        """ x0 : initial position
            velocity: (+=right, -=left)
            measurement_var: variance in measurement m^2
            process_var: variance in process (m/s)^2
        """
        self.x = x0
        self.velocity = velocity
        self.meas_std = sqrt(measurement_var)
        self.process_std = sqrt(process_var)

    def move(self, dt=1.0):
        """Compute new position of the dog in dt seconds."""
        # multiplying a standard normal distribution sample by std deviation creates mean 0 and std Gaussian
        dx = self.velocity + np.random.randn() * self.process_std
        self.x += dx * dt

    def sense_position(self):
        """ Returns measurement of new position in meters."""
        measurement = self.x + np.random.randn() * self.meas_std
        return measurement

    def move_and_sense(self):
        """ Move dog, and return measurement of new position in meters"""
        self.move()
        return self.sense_position()
