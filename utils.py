import numpy as np
import matplotlib.pyplot as plt


def plot_data(estimates, measurements, truth):
    estimates = np.array(estimates)
    measurements = np.array(measurements)
    truth = np.array(truth)

    plt.plot(truth, ls="-", color="k", label="Truth")
    plt.plot(estimates, ls=":", color="b", label="Estimates")
    plt.scatter(np.array([range(len(truth))]), measurements, marker="o", color="k", label="Measurements")
    plt.grid()
    plt.legend()
    plt.show()
