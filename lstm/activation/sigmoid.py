"""
Sigmoid actication function module.

Read more https://en.wikipedia.org/wiki/Sigmoid_function
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def sigmoid(
    x: Union[np.float64, NDArray[np.float64]],
) -> Union[np.float64, NDArray[np.float64]]:
    """
    Sigmoid activation function.

    Boundaries:
        Domain (-inf, inf)
        Range (0, 1)
        Codomain (0, 1)

    Args:
        x (np.float64, NDArray[np.float64]) - The input value as a
            single value or an array of values.
    Returns:
        np.float64, NDArray[np.float64] - The sigmoid of the input value.
    """
    return 1 / (1 + np.exp(-x))
