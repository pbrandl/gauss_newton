from typing import Callable, List

import numpy as np

METHODS = ("forward", "backward", "central")


class GaussNewton:
    """Gauss-Newton optimizer using finite-difference Jacobians."""

    def __init__(self, error_fun: Callable, max_iteration: int = 10, tolerance: float = 1e-2,
                 method: str = "forward"):
        if max_iteration < 1:
            raise ValueError("max_iteration must be >= 1.")
        if method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}.")

        self.error_fun = error_fun
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.method = method
        self.error_history: List[float] = []

    def fit(self, model: Callable, param: np.ndarray, target: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """
        Fit the model to the target values, starting from some initial parameters.

        :param model: Callable ``model(param, samples)`` returning predictions.
        :param param: Initial parameters (column vector). Not modified.
        :param target: Target values.
        :param samples: Input data passed to the model.
        :return: Optimized parameters.
        """
        param = np.array(param, dtype=float)
        self.error_history = [self.error_fun(target, model(param, samples))]

        for i in range(self.max_iteration):
            residuals = (target - model(param, samples)).reshape(-1, 1)
            J = self.jacobian(model, param, samples)
            param -= np.linalg.solve(J @ J.T, J @ residuals)

            error = self.error_fun(target, model(param, samples))
            self.error_history.append(error)
            if abs(self.error_history[-2] - error) <= self.tolerance:
                print(f"Converged at iteration {i}.")
                break

        print(f"Error: {self.error_history[-1]:.4f}")
        return param

    def jacobian(self, f: Callable, param: np.ndarray, x: np.ndarray, step: float = 1e-3) -> np.ndarray:
        """
        Compute the Jacobian of the residuals r = y - f(param, x), i.e. J_ij = d(r_j)/d(param_i).

        :param f: Model function.
        :param param: Parameters of the function (column vector).
        :param x: Input data to the function.
        :param step: Step width for finite differences.
        :return: The Jacobian matrix of shape (n_parameters, n_samples).
        """
        n = param.shape[0]
        J = []
        for disturb in np.eye(n):
            h = step * disturb.reshape(param.shape)
            if self.method == "forward":
                fd = (f(param, x) - f(param + h, x)) / step
            elif self.method == "backward":
                fd = (f(param - h, x) - f(param, x)) / step
            else:  # central
                fd = (f(param - h, x) - f(param + h, x)) / (2 * step)
            J.append(fd.flatten())

        return np.asarray(J)
