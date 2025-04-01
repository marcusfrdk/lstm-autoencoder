"""Unit tests for the ReLU activation function."""

import numpy as np
from numpy.testing import assert_almost_equal, assert_array_equal

from lstm.activation.relu import relu


def test_relu_positive_scalar():
    """Test that ReLU returns the same value for positive inputs."""
    x1 = 5.0
    e1 = 5.0
    assert relu(x1) == e1

    x2 = 0.5
    e2 = 0.5
    assert relu(x2) == e2

    x3 = 100.0
    e3 = 100.0
    assert relu(x3) == e3


def test_relu_negative_scalar():
    """Test that ReLU returns 0 for negative inputs."""
    x1 = -1.0
    e1 = 0.0
    assert relu(x1) == e1

    x2 = -100.0
    e2 = 0.0
    assert relu(x2) == e2

    x3 = -0.5
    e3 = 0.0
    assert relu(x3) == e3


def test_relu_zero():
    """Test that ReLU returns 0 for zero input."""
    x = 0.0
    e = 0.0
    assert relu(x) == e


def test_relu_array():
    """Test ReLU with array inputs."""
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    e = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    assert_array_equal(relu(x), e)


def test_relu_2d_array():
    """Test ReLU with 2D array inputs."""
    x = np.array([[-1.0, 2.0], [3.0, -4.0]])
    e = np.array([[0.0, 2.0], [3.0, 0.0]])
    assert_array_equal(relu(x), e)


def test_relu_large_values():
    """Test ReLU with large positive and negative values."""
    x1 = 1e10
    e1 = 1e10
    assert relu(x1) == e1

    x2 = -1e10
    e2 = 0.0
    assert relu(x2) == e2


def test_relu_small_values():
    """Test ReLU with values close to zero."""
    x1 = 1e-10
    e1 = 1e-10
    assert_almost_equal(relu(x1), e1)

    x2 = -1e-10
    e2 = 0.0
    assert_almost_equal(relu(x2), e2)


def test_relu_float_precision():
    """Test ReLU with floating point precision considerations."""
    x = np.array([1e-8, -1e-8])
    e = np.array([1e-8, 0.0])
    assert_almost_equal(relu(x), e)
