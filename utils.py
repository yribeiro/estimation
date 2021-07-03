import numpy as np
import matplotlib.pyplot as plt


def plot_data(estimates, predictions, measurements, truth):
    # all data is from t1 onwards .. except truth
    estimates = np.array(estimates)
    measurements = np.array(measurements)
    predictions = np.array(predictions)
    truth = np.array(truth)
    time = np.array(range(1, len(measurements)+1))

    plt.plot(truth, ls="-", color="k", label="Truth")
    plt.plot(time, estimates, ls=":", color="b", label="Estimates")
    plt.scatter(time, measurements, marker="o", color="k", label="Measurements")
    plt.scatter(time, predictions, marker="^", color="r", label="Predictions")
    plt.grid()
    plt.legend()
    plt.show()
