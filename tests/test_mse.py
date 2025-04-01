"""Unit tests for Mean Squared Error (MSE) loss function."""

import numpy as np
import pytest

from lstm.loss.mse import mse


def test_mse_simple_case():
    """Test MSE with a simple case where result is known."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    e = 5 / 3
    assert np.isclose(mse(y_true, y_pred), e)


def test_mse_perfect_prediction():
    """Test MSE when predictions are perfect."""
    y_true = np.array([1.5, 2.5, 3.5])
    y_pred = np.array([1.5, 2.5, 3.5])
    e = 0.0
    assert np.isclose(mse(y_true, y_pred), e)


def test_mse_2d_arrays():
    """Test MSE with 2D arrays."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 1.0], [1.0, 1.0]])
    e = 3.5
    assert np.isclose(mse(y_true, y_pred), e)


def test_mse_negative_values():
    """Test MSE with negative values."""
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    e = 56 / 3
    assert np.isclose(mse(y_true, y_pred), e)


def test_mse_floating_point():
    """Test MSE with floating point values."""
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred = np.array([0.15, 0.25, 0.35])
    e = 0.0025
    assert np.isclose(mse(y_true, y_pred), e)


def test_mse_shape_mismatch():
    """Test MSE raises error with incompatible shapes."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        mse(y_true, y_pred)


def test_mse_empty_arrays():
    """Test MSE with empty arrays."""
    y_true = np.array([], dtype=np.float64)
    y_pred = np.array([], dtype=np.float64)

    with pytest.raises(ValueError):
        mse(y_true, y_pred)
