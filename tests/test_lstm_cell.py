"""Unit tests for the LSTMCell class."""

# pylint: disable=redefined-outer-name

import numpy as np
import pytest

from lstm.classes.lstm_cell import LSTMCell


@pytest.fixture
def lstm_cell():
    """Create a basic LSTM cell with fixed dimensions for testing."""
    np.random.seed(42)
    return LSTMCell(input_size=3, hidden_size=4)


def test_initialization(lstm_cell):
    """Test that LSTM cell initializes with correct shapes and values."""
    assert lstm_cell.input_size == 3
    assert lstm_cell.hidden_size == 4
    assert lstm_cell.concat_size == 7

    assert lstm_cell.W_f.shape == (4, 7)
    assert lstm_cell.W_i.shape == (4, 7)
    assert lstm_cell.W_o.shape == (4, 7)
    assert lstm_cell.W_c.shape == (4, 7)

    assert lstm_cell.b_f.shape == (4, 1)
    assert lstm_cell.b_i.shape == (4, 1)
    assert lstm_cell.b_o.shape == (4, 1)
    assert lstm_cell.b_c.shape == (4, 1)

    assert np.all(lstm_cell.b_f == 0)
    assert np.all(lstm_cell.b_i == 0)
    assert np.all(lstm_cell.b_o == 0)
    assert np.all(lstm_cell.b_c == 0)


def test_forward_pass_shapes(lstm_cell):
    """Test that forward pass returns correct shapes."""
    x = np.random.randn(3, 1)
    h_prev = np.random.randn(4, 1)
    c_prev = np.random.randn(4, 1)

    h_next, c_next = lstm_cell.forward(x, h_prev, c_prev)

    assert h_next.shape == (4, 1)
    assert c_next.shape == (4, 1)


def test_forward_pass_deterministic(lstm_cell):
    """Test that forward pass gives deterministic results for fixed inputs."""
    x = np.ones((3, 1))
    h_prev = np.ones((4, 1))
    c_prev = np.ones((4, 1))

    h1, c1 = lstm_cell.forward(x, h_prev, c_prev)
    h2, c2 = lstm_cell.forward(x, h_prev, c_prev)

    np.testing.assert_array_equal(h1, h2)
    np.testing.assert_array_equal(c1, c2)
