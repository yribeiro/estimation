import matplotlib.pyplot as plt
import filterpy.stats as stats

from typing import NamedTuple


class Gaussian(NamedTuple):
    mean: float
    variance: float

    def __repr__(self):
        return f"N(mu: {self.mean:.2f}, sigma: {self.variance:.2f})"


# region predict
def predict(pos, movement):
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

if __name__ == "__main__":
    g = Gaussian(mean=10, variance=1)
    print(g)
    stats.plot_gaussian_pdf(g.mean, g.variance)
    plt.grid()
    plt.show()
