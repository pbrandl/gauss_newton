from typing import Callable
import numpy as np


class GaussNewton(object):
    def __init__(self, error_fun: Callable, max_iteration: int = 10, tolerance: float = 1e-2, method: str = 'forward'):
        assert max_iteration >= 1, "Maximum number of iterations must be set to greater equal 1."

        self.error_fun = error_fun
        self.max_iteration = max_iteration
        self.tolerance = tolerance
        self.method = method

        self.rmse_history = []

    def fit(self, model: Callable, param: np.ndarray, target: np.ndarray, samples: np.ndarray) -> np.ndarray:
        """
        Fit the model to y given some inital parameters.

        :param model: The model to be fit. Callable function.
        :param param: Parameters of the model.
        :param target: Target values.
        :return: Optimized parameters.
        """

        for i in range(self.max_iteration):
            # Compute RMSE
            prediction = model(param, samples)
            rmse0 = self.error_fun(prediction, target)
            self.rmse_history.append(rmse0)

            # Optimize parameters
            residuals = (target - model(param, samples)).reshape(-1, 1)
            J = self.jacobian(model, param, samples)
            param -= np.linalg.inv(J @ J.T) @ J @ residuals

            # Check if convergence criterium is reached
            rmse1 = self.error_fun(prediction, target)
            if abs(rmse0 - rmse1) <= self.tolerance:
                print("Non-changing: Converged at iteration", i)
                break

            self.rmse_history.append(rmse0)
        print('RMSE: {:.2f}'.format(self.rmse_history[-1]))
        return param

    def jacobian(self, f, param, x, step: float = 1e-03):
        """
        Compute the Jacobian matrix as J_ij = d(r_i)/d(param_j).

        :param f: Callable function.
        :param param: Parameter of the function.
        :param x: Input data to the function
        :param step: Step width for finite differences.
        :return: The Jacobian matrix of shape n_parameter x n_x
        """
        n = param.shape[0]
        J = []
        for i, disturb in enumerate(np.eye(n).reshape(n, 1, n)):
            if self.method == 'forward':
                fd = (f(param, x) - f(param + step * disturb.T, x)) / step
            elif self.method == 'backward':
                fd = (f(param - step * disturb.T, x) - f(param, x)) / step
            elif self.method == 'central':
                fd = (f(param - step * disturb.T, x) - f(param + step * disturb.T, x)) / (2 * step)
            else:
                raise Exception("Given method not one of foward, backward or central.")

            J.append(fd.flatten())

        return np.asarray(J)
