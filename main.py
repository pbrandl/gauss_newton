import matplotlib.pyplot as plt
import numpy as np

from gaussnewton import GaussNewton
from polynomial import Polynomial

np.set_printoptions(precision=2)


def add_noise(x: np.ndarray, mu: float = 0., sigma: float = 1.) -> np.ndarray:
    """Add Gaussian noise to the array."""
    return x + np.random.normal(mu, sigma, x.shape)


def rmse(target: np.ndarray, prediction: np.ndarray) -> float:
    """Root mean squared error."""
    return float(np.sqrt(((target - prediction) ** 2).mean()))


def main():
    # Interval to sample from
    lower, upper = -1.5, 1.6
    space = np.arange(lower, upper, 0.1)
    samples = np.random.uniform(lower, upper, size=(1, 10))

    # True and initial parameters (polynomial coefficients)
    true_param = np.random.uniform(-2, 2, size=(7, 1))
    init_param = np.random.uniform(-4, 4, size=(6, 1))

    print("Observations:", samples.size)
    print("Initial parameters:", init_param.flatten())
    print("True parameters:", true_param.flatten())

    # Generate noisy observations and fit the model
    model = Polynomial()
    y_noise = add_noise(model(true_param, samples), mu=0., sigma=1.)
    gn = GaussNewton(rmse, max_iteration=40, tolerance=1e-4, method="central")
    param_opt = gn.fit(model, init_param, y_noise, samples)

    print("Optimized model:\n", model.to_str(param_opt))
    print("True model:\n", model.to_str(true_param))

    # Plot the resulting model
    fig, ax = plt.subplots()
    ax.plot(space, model(param_opt, space), label="Fit", alpha=0.7)
    ax.plot(space, model(true_param, space), label="Truth", alpha=0.7)
    ax.scatter(samples, y_noise, label="Noisy samples", color="orange", alpha=0.9)
    ax.legend()
    plt.show()


if __name__ == "__main__":
    main()
