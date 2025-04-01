"""Unit tests for the sigmoid activation function."""

import numpy as np

from lstm.activation.sigmoid import sigmoid


def test_sigmoid_zero():
    """Test sigmoid function at x = 0."""
    x = 0
    e = 0.5
    assert sigmoid(x) == e


def test_sigmoid_positive():
    """Test sigmoid function with a positive input."""
    x = 2
    e = 0.8807970779778823
    assert np.isclose(sigmoid(x), e)


def test_sigmoid_negative():
    """Test sigmoid function with a negative input."""
    x = -2
    e = 0.11920292202211755
    assert np.isclose(sigmoid(x), e)


def test_sigmoid_large_values():
    """Test sigmoid function with large values."""
    x1 = 100
    e1 = 1.0
    x2 = -100
    e2 = 0.0
    assert np.isclose(sigmoid(x1), e1)
    assert np.isclose(sigmoid(x2), e2)


def test_sigmoid_array():
    """Test sigmoid function with numpy array input."""
    x = np.array([0, 1, -1])
    e = np.array([0.5, 0.7310585786300049, 0.2689414213699951])
    np.testing.assert_array_almost_equal(sigmoid(x), e)
