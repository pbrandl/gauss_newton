"""Example models that can be fitted with :class:`gauss_newton.GaussNewton`."""

import numpy as np
from numpy.typing import ArrayLike, NDArray


class Polynomial:
    """Polynomial model ``f(params, x) = params[0] + params[1] * x + ... + params[n] * x**n``.

    The degree is given by the number of parameters, so the same instance can be used
    for polynomials of any degree.
    """

    def __call__(self, params: ArrayLike, x: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the polynomial with coefficients ``params`` at ``x``."""
        coefficients = np.ravel(np.asarray(params, dtype=float))
        x = np.asarray(x, dtype=float)
        result = np.zeros_like(x)
        for coefficient in coefficients[::-1]:  # Horner's scheme
            result = result * x + coefficient
        return result

    @staticmethod
    def to_string(params: ArrayLike, precision: int = 2) -> str:
        """Format the polynomial, e.g. ``"1.00 - 2.00x + 0.50x^2"``."""
        coefficients = np.ravel(np.asarray(params, dtype=float))
        terms = []
        for power, coefficient in enumerate(coefficients):
            variable = "" if power == 0 else "x" if power == 1 else f"x^{power}"
            value = f"{abs(coefficient):.{precision}f}{variable}"
            if power == 0:
                terms.append(f"-{value}" if coefficient < 0 else value)
            else:
                terms.append(f"{'-' if coefficient < 0 else '+'} {value}")
        return " ".join(terms)
