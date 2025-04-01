"""
Mean Squared Error loss function.

Read more https://en.wikipedia.org/wiki/Mean_squared_error
"""

import numpy as np
from numpy.typing import NDArray

from lstm.typing.number import NDNumber


def mse(y_true: NDArray[NDNumber], y_pred: NDArray[NDNumber]) -> np.float64:
    """
    Mean Squared Error loss function.

    Args:
        y_true (NDArray[NDNumber]): True labels.
        y_pred (NDArray[NDNumber]): Predicted labels.

    Returns:
        np.float64: Mean Squared Error.
    """
    if y_true.shape != y_pred.shape:
        raise ValueError("Shapes of y_true and y_pred must match.")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_true and y_pred must not be empty.")
    return np.mean(np.square(y_true - y_pred))
