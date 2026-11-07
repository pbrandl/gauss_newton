"""Error metrics used to monitor convergence."""

import numpy as np
from numpy.typing import ArrayLike


def rmse(target: ArrayLike, prediction: ArrayLike) -> float:
    """Return the root mean squared error between ``target`` and ``prediction``."""
    diff = np.asarray(target, dtype=float) - np.asarray(prediction, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))
