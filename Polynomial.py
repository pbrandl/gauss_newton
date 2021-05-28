import numpy as np


class Polynomial:
    """
    Class of simple polynomial arbitraritly parameterized.
    """

    def __call__(self, parameters, x):
        return self.evaluate(parameters, x)

    @staticmethod
    def evaluate(parameters, x):
        """
        Returns the solution of a polynomial:
        a + b*x + c*x^2 ... d*x^n, where a, b, c, d are coeffs.
        """
        return np.asarray([p * x ** i for i, p in enumerate(parameters)]).sum(axis=0)

    @staticmethod
    def to_str(parameters):
        return " + ".join(["{:0.1f}x^{}".format(p.item(), i) for i, p in enumerate(parameters)])
