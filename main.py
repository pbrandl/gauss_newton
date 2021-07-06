import numpy as np
import matplotlib.pyplot as plt
from GaussNewton import GaussNewton
from Polynomial import Polynomial

np.set_printoptions(precision=2)


def add_noise(x: np.ndarray, mu=0., sigma=1.):
    """
    Add Gaussian noise to the array.
    """
    return x + np.random.normal(mu, sigma, x.shape)


def rmse(target, prediction):
    """
    Returns the mean squared error.
    """
    return ((target - prediction) ** 2).mean() / 2


if __name__ == "__main__":
    # Define space where to sample from
    lower, upper = -1.5, 1.6  # upper lower bound
    space = np.mgrid[lower:upper:0.1]
    samples = np.random.uniform(lower, upper, size=(1, 10))
    sample_x = samples
    x = space

    # Define target and initial parameter
    x_param = np.random.uniform(-4, 4, size=(6, 1))
    y_param = np.random.uniform(-2, 2, size=(7, 1))

    print("Observations:", len(sample_x))
    print("Initital parameters:", x_param.flatten())
    print("True parameters", y_param.flatten())

    # Set up a model and fit
    model = Polynomial()
    y_noise = add_noise(model(y_param, samples), mu=0., sigma=1.)
    gn = GaussNewton(rmse, max_iteration=40, tolerance=1e-4, method='central')
    param_opt = gn.fit(model, x_param, y_noise, samples)

    # Plot the resulting model
    fig = plt.figure()
    ax = fig.add_subplot(111, )
    ax.plot(x, model(param_opt, space), label='Fit', alpha=0.7)
    ax.plot(x, model(y_param, space), label='Truth', alpha=0.7)
    ax.scatter(sample_x, y_noise, label="Noisy Samples", color='orange', alpha=0.9)
    plt.legend()
    plt.show()

    print("Optimized model:\n", model.to_str(param_opt))
    print("Objective parameters:\n", model.to_str(y_param))
