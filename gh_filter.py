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
    estimates, predictions = [], []

    # loop over all measurements from t1
    for z in data:
        # predict what the value will be at t1
        x_pred = x_est + (dx * dt)
        predictions.append(x_pred)
        dx = dx

        # get the residual between the measurement at t1 and the prediction t1
        residual = z - x_pred

        # update the second order and state prediction from above to get the estimate at t1
        dx = dx + h * residual / dt
        x_est = x_pred + g * residual

        estimates.append(x_est)

    return estimates, predictions


if __name__ == "__main__":
    import numpy as np
    from utils import plot_data

    # model initial conditions
    x0 = 160
    dx0 = 1
    count = 20

    # this is a straight line and the filter is trying to track the progression
    truth = [x0, *[x0 + dx0 * i for i in range(1, count)]]
    # measurements are taken every second
    measurements = [t + np.random.normal(0, 2.5) for t in truth]
    estimates, predictions = g_h_filter(
        data=measurements,
        x0=x0,
        dx=dx0,
        g=0.6,
        h=1/3,
        dt=dx0
    )

    plot_data(estimates, predictions, measurements, truth)
