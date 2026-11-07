import numpy as np


class Polynomial:
    """Polynomial with arbitrary degree, defined by its coefficients."""

    def __call__(self, parameters, x):
        return self.evaluate(parameters, x)

    @staticmethod
    def evaluate(parameters, x):
        """
        Evaluate a0 + a1*x + a2*x^2 + ... + an*x^n, where a0..an are the parameters.
        """
        return np.asarray([p * x ** i for i, p in enumerate(parameters)]).sum(axis=0)

    @staticmethod
    def to_str(parameters):
        return " + ".join(f"{p.item():0.2f}x^{i}" for i, p in enumerate(parameters))
