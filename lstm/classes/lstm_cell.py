"""An LSTM (Long Short-Term Memory) cell implementation."""

# pylint: disable=invalid-name,too-many-instance-attributes

from typing import Tuple

import numpy as np

from lstm.activation.sigmoid import sigmoid
from lstm.activation.tanh import tanh
from lstm.typing.number import NDInt


class LSTMCell:
    """LSTM (Long Short-Term Memory) cell class."""

    def __init__(self, input_size: NDInt, hidden_size: NDInt):
        """
        Initialize the LSTM cell with random weights and biases.

        Names:
            - W_*: Weights for the forget, input, output, and cell gates.
            - b_*: Biases for the forget, input, output, and cell gates.
            - input_size: Size of the input features.
            - hidden_size: Size of the hidden state.
            - concat_size: Size of the concatenated input and hidden state.
        Args:
            input_size (NDInt): Size of the input features.
            hidden_size (NDInt): Size of the hidden state.
        Returns:
            None
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.concat_size = input_size + hidden_size

        # Forget gate
        self.W_f = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.b_f = np.zeros((self.hidden_size, 1))

        # Input gate
        self.W_i = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.b_i = np.zeros((self.hidden_size, 1))

        # Output gate
        self.W_o = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.b_o = np.zeros((self.hidden_size, 1))

        # Cell gate
        self.W_c = np.random.randn(self.hidden_size, self.concat_size) * 0.1
        self.b_c = np.zeros((self.hidden_size, 1))

    def __str__(self):
        return " ".join(
            [
                "<LSTMCell",
                f" input={self.input_size}",
                f" hidden={self.hidden_size}",
                f" forget=(({self.W_f.shape}), {self.b_f.shape})",
                f" input=(({self.W_i.shape}), {self.b_i.shape})",
                f" output=(({self.W_o.shape}), {self.b_o.shape})",
                f" cell=(({self.W_c.shape}), {self.b_c.shape})",
                ">",
            ]
        )

    def __repr__(self):
        return f"<LSTMCell input={self.input_size} hidden={self.hidden_size}>"

    def forward(
        self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through the LSTM cell.

        Args:
            x (np.ndarray): Input data.
            h_prev (np.ndarray): Previous hidden state.
            c_prev (np.ndarray): Previous cell state.
        Returns:
            tuple: Tuple containing the next hidden state and cell state.
        """
        concat = np.vstack((h_prev, x))
        f_gate = sigmoid(np.dot(self.W_f, concat) + self.b_f)
        i_gate = sigmoid(np.dot(self.W_i, concat) + self.b_i)
        o_gate = sigmoid(np.dot(self.W_o, concat) + self.b_o)
        c_gate = tanh(np.dot(self.W_c, concat) + self.b_c)
        c_next = f_gate * c_prev + i_gate * c_gate
        h_next = o_gate * tanh(c_next)
        return h_next, c_next
