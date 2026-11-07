"""Gauss-Newton least-squares optimization with finite-difference Jacobians."""

from gauss_newton.metrics import rmse
from gauss_newton.models import Polynomial
from gauss_newton.optimizer import FitResult, GaussNewton, finite_difference_jacobian

__all__ = [
    "FitResult",
    "GaussNewton",
    "Polynomial",
    "finite_difference_jacobian",
    "rmse",
]

__version__ = "0.1.0"
