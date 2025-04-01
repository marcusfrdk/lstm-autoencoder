"""
Rectified Linear Unit (ReLU) activation function.

Read more https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def relu(
    x: Union[np.float64, NDArray[np.float64]],
) -> Union[np.float64, NDArray[np.float64]]:
    """
    Rectified Linear Unit (ReLU) activation function.

    Boundaries:
        Domain (-inf, inf)
        Range (0, inf)
        Codomain (0, inf)
    Args:
        x (np.float64, NDArray[np.float64]) - The input value as a
            single value or an array of values.
    Returns:
        np.float64, NDArray[np.float64] - The ReLU of the input value.
    """
    return np.maximum(0, x)
