"""Number type that combines NumPy ints and floats."""

from typing import Union

from numpy import float32, float64, int32, int64

NDNumber = Union[float64, float32, int64, int32]
NDInt = Union[int32, int64]
NDFloat = Union[float32, float64]
