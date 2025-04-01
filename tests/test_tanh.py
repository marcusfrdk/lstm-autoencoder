"""Unit tests for the tanh activation function."""

import numpy as np

from lstm.activation.tanh import tanh


def test_tanh_single_value():
    """Test tanh with single float values."""
    # Test with zero
    x1 = 0.0
    e1 = 0.0
    assert tanh(x1) == e1

    # Test with positive value
    x2 = 1.0
    e2 = 0.7615941559557649
    assert np.isclose(tanh(x2), e2)

    # Test with negative value
    x3 = -1.0
    e3 = -0.7615941559557649
    assert np.isclose(tanh(x3), e3)


def test_tanh_array():
    """Test tanh with numpy arrays."""
    # Test with array of values
    x = np.array([0.0, 1.0, -1.0])
    e = np.array([0.0, 0.7615941559557649, -0.7615941559557649])
    assert np.allclose(tanh(x), e)


def test_tanh_boundaries():
    """Test tanh approaches its boundaries (-1, 1) for large values."""
    # Large positive value
    x1 = 10.0
    e1 = 1.0
    assert np.isclose(tanh(x1), e1, atol=1e-4)

    # Large negative value
    x2 = -10.0
    e2 = -1.0
    assert np.isclose(tanh(x2), e2, atol=1e-4)
