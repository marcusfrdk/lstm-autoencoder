"""Unit tests for Mean Absolute Error (MAE) loss function."""

import numpy as np
import pytest

from lstm.loss.mae import mae


def test_mae_simple_case():
    """Test MAE with a simple case where result is known."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 1.0, 1.0])
    e = 1.0
    assert np.isclose(mae(y_true, y_pred), e)


def test_mae_perfect_prediction():
    """Test MAE when predictions are perfect."""
    y_true = np.array([1.5, 2.5, 3.5])
    y_pred = np.array([1.5, 2.5, 3.5])
    e = 0.0
    assert np.isclose(mae(y_true, y_pred), e)


def test_mae_2d_arrays():
    """Test MAE with 2D arrays."""
    y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
    y_pred = np.array([[1.0, 1.0], [1.0, 1.0]])
    e = 1.5
    assert np.isclose(mae(y_true, y_pred), e)


def test_mae_negative_values():
    """Test MAE with negative values."""
    y_true = np.array([-1.0, -2.0, -3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    e = 4.0
    assert np.isclose(mae(y_true, y_pred), e)


def test_mae_floating_point():
    """Test MAE with floating point values."""
    y_true = np.array([0.1, 0.2, 0.3])
    y_pred = np.array([0.15, 0.25, 0.35])
    e = 0.05
    assert np.isclose(mae(y_true, y_pred), e)


def test_mae_shape_mismatch():
    """Test MAE raises error with incompatible shapes."""
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0])

    with pytest.raises(ValueError):
        mae(y_true, y_pred)


def test_mae_empty_arrays():
    """Test MAE with empty arrays."""
    y_true = np.array([])
    y_pred = np.array([])

    with pytest.raises(ValueError):
        mae(y_true, y_pred)
