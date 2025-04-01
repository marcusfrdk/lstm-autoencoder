"""
Tanh activation function.

Read more https://en.wikipedia.org/wiki/Hyperbolic_functions#Hyperbolic_tangent
"""

from typing import Union

import numpy as np
from numpy.typing import NDArray


def tanh(
    x: Union[np.float64, NDArray[np.float64]],
) -> Union[np.float64, NDArray[np.float64]]:
    """
    Hyperbolic tangent activation function.

    Boundaries:
        Domain (-inf, inf)
        Range (-1, 1)
        Codomain (-1, 1)

    Args:
        x (np.float64, NDArray[np.float64]) - The input value as a
            single value or an array of values.
    Returns:
        np.float64, NDArray[np.float64] - The tanh of the input value.
    """
    return np.tanh(x)
