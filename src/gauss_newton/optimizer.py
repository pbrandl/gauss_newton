"""Gauss-Newton optimizer for nonlinear least-squares problems."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from numpy.typing import ArrayLike, NDArray

from gauss_newton.metrics import rmse

logger = logging.getLogger(__name__)

Model = Callable[[NDArray[np.float64], NDArray[np.float64]], ArrayLike]
"""A model ``model(params, x)`` returning one prediction per observation."""

ErrorFunction = Callable[[NDArray[np.float64], NDArray[np.float64]], float]
"""An error metric ``error_fun(target, prediction)`` used to monitor convergence."""

DifferenceMethod = Literal["forward", "backward", "central"]
"""Finite-difference scheme used to approximate the Jacobian."""

_METHODS: tuple[str, ...] = get_args(DifferenceMethod)


@dataclass(frozen=True)
class FitResult:
    """Outcome of :meth:`GaussNewton.fit`.

    Attributes:
        params: Optimized parameters, with the same shape as the initial parameters.
        converged: Whether the change in error fell below the tolerance.
        n_iterations: Number of Gauss-Newton steps performed.
        history: Error before the first step followed by the error after each step.
    """

    params: NDArray[np.float64]
    converged: bool
    n_iterations: int
    history: list[float]

    @property
    def error(self) -> float:
        """Error of the final parameters."""
        return self.history[-1]


def finite_difference_jacobian(
    model: Model,
    params: ArrayLike,
    x: ArrayLike,
    step: float = 1e-6,
    method: DifferenceMethod = "central",
) -> NDArray[np.float64]:
    """Approximate the Jacobian of ``model`` with respect to its parameters.

    Args:
        model: Callable ``model(params, x)``.
        params: Parameters at which the Jacobian is evaluated.
        x: Input data passed to the model.
        step: Step width for the finite differences.
        method: ``"forward"``, ``"backward"`` or ``"central"`` differences.

    Returns:
        Matrix ``J`` of shape ``(n_observations, n_params)`` with
        ``J[i, j] = d model(params, x)[i] / d params[j]``, where both the model output
        and the parameters are flattened.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}.")

    params = np.asarray(params, dtype=float)
    x = np.asarray(x, dtype=float)

    def evaluate(p: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.ravel(np.asarray(model(p, x), dtype=float))

    base = evaluate(params)
    columns = []
    for unit_vector in np.eye(params.size):
        offset = (step * unit_vector).reshape(params.shape)
        if method == "forward":
            column = (evaluate(params + offset) - base) / step
        elif method == "backward":
            column = (base - evaluate(params - offset)) / step
        else:
            column = (evaluate(params + offset) - evaluate(params - offset)) / (2 * step)
        columns.append(column)

    return np.column_stack(columns)


class GaussNewton:
    """Gauss-Newton optimizer using finite-difference Jacobians.

    Minimizes the sum of squared residuals ``y - model(params, x)``. Each step solves the
    linearized least-squares problem ``J @ delta ≈ residuals`` and updates
    ``params += delta``. Iteration stops once the error changes by at most ``tolerance``.

    Args:
        error_fun: Metric ``error_fun(target, prediction)`` used to monitor convergence.
        max_iterations: Maximum number of Gauss-Newton steps.
        tolerance: Absolute change in error at which the optimization is considered
            converged.
        method: Finite-difference scheme for the Jacobian.
        step: Step width for the finite differences.
    """

    def __init__(
        self,
        error_fun: ErrorFunction = rmse,
        max_iterations: int = 100,
        tolerance: float = 1e-8,
        method: DifferenceMethod = "central",
        step: float = 1e-6,
    ) -> None:
        if max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1, got {max_iterations}.")
        if tolerance < 0:
            raise ValueError(f"tolerance must be non-negative, got {tolerance}.")
        if method not in _METHODS:
            raise ValueError(f"method must be one of {_METHODS}, got {method!r}.")
        if step <= 0:
            raise ValueError(f"step must be positive, got {step}.")

        self.error_fun = error_fun
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.method = method
        self.step = step

    def fit(self, model: Model, x: ArrayLike, y: ArrayLike, initial_params: ArrayLike) -> FitResult:
        """Fit ``model`` to the observations ``(x, y)``.

        Args:
            model: Callable ``model(params, x)`` returning one prediction per element of
                ``y``.
            x: Input data passed to the model.
            y: Observed target values.
            initial_params: Starting point of the optimization. Not modified.

        Returns:
            The optimized parameters together with convergence information.
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        params = np.array(initial_params, dtype=float)  # copy, keeps the caller's array intact

        def evaluate(p: NDArray[np.float64]) -> tuple[NDArray[np.float64], float]:
            prediction = np.asarray(model(p, x), dtype=float)
            if prediction.size != y.size:
                raise ValueError(
                    f"model returned {prediction.size} predictions for {y.size} observations."
                )
            prediction = prediction.reshape(y.shape)
            return np.ravel(y - prediction), self.error_fun(y, prediction)

        residuals, error = evaluate(params)
        history = [error]
        converged = False
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            jacobian = finite_difference_jacobian(model, params, x, self.step, self.method)
            delta, *_ = np.linalg.lstsq(jacobian, residuals, rcond=None)
            params = params + delta.reshape(params.shape)

            residuals, error = evaluate(params)
            history.append(error)
            logger.debug("Iteration %d: error %.6g", iteration, error)

            if not np.isfinite(error):
                logger.warning("Error became non-finite at iteration %d; stopping.", iteration)
                break
            if abs(history[-2] - error) <= self.tolerance:
                converged = True
                break

        if converged:
            logger.info("Converged after %d iterations with error %.6g.", iteration, error)
        elif np.isfinite(error):
            logger.warning("Did not converge within %d iterations.", self.max_iterations)

        return FitResult(
            params=params, converged=converged, n_iterations=iteration, history=history
        )
