"""Recurrent sequence models for VR cybersickness prediction."""

import torch
import torch.nn as nn


class GRUPredictor(nn.Module):
    """Two-layer GRU for minute-level cybersickness score regression (Task 1).

    Args:
        input_size: Number of input features per time step.
        hidden: GRU hidden dimension.
    """

    def __init__(self, input_size: int, hidden: int = 64):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, 2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Input tensor of shape (batch, seq_len, input_size).
        Returns:
            Predicted scores of shape (batch,).
        """
        _, h_n = self.gru(x)
        return self.head(h_n[-1]).squeeze(-1)


class SequenceToOneLSTM(nn.Module):
    """Two-layer LSTM for session-level SSQ Total Score regression (Task 2).

    Handles variable-length sequences via pack_padded_sequence.

    Args:
        input_size: Number of input features per time step.
        hidden: LSTM hidden dimension.
    """

    def __init__(self, input_size: int = 12, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, 2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(
            nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Padded input tensor of shape (batch, max_seq_len, input_size).
            lengths: Actual sequence lengths of shape (batch,).
        Returns:
            Predicted SSQ scores of shape (batch,).
        """
        packed = nn.utils.rnn.pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        return self.head(h_n[-1]).squeeze(-1)
