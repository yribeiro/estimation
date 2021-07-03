def g_h_filter(data, x0, dx, g, h, dt):
    """
    Performs g-h filter on 1 state variable with a fixed g and h.

    'data' contains the data to be filtered.
    'x0' is the initial value for our state variable
    'dx' is the initial change rate for our state variable
    'g' is the g-h's g scale factor
    'h' is the g-h's h scale factor
    'dt' is the length of the time step
    """
    # set initial conditions
    x_est = x0
    estimates = [x_est]

    # loop over all measurements
    for z in data:
        # predict
        x_pred = x_est + (dx * dt)
        dx = dx

        # get the residual between the measurement and the prediction
        residual = z - x_pred

        # update the second order and state prediction from above
        dx = dx + h * (residual / dt)
        x_est = x_pred + g * residual

        estimates.append(x_est)

    return estimates


if __name__ == "__main__":
    import numpy as np
    from utils import plot_data

    truth = [160 + i for i in range(10)]
    measurements = [t + np.random.normal(0, 0.5) for t in truth]

    plot_data([], measurements, truth)
