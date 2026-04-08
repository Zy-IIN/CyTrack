"""
Model definitions for the VR Cybersickness Dual-Task Benchmark.

Task 1 (minute-level):
    - GRUPredictor           : 2-layer GRU baseline
    - ModalityGate           : SQI-based hard gating per modality
    - MultimodalFusion       : Dual-branch validity-gated fusion network

Task 2 (session-level):
    - SequenceToOneLSTM      : 2-layer LSTM with pack_padded_sequence
    - ModalityEncoder        : Per-timestep modality encoder with validity gating
    - SequenceToOnePredictor : Dual-branch validity-gated LSTM fusion
"""

import torch
import torch.nn as nn


# ======================================================================
# Task 1 Models
# ======================================================================

class GRUPredictor(nn.Module):
    """2-layer GRU for minute-level score regression (Task 1)."""

    def __init__(self, input_size, hidden=64, task='reg'):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, 2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x):
        _, h_n = self.gru(x)
        return self.head(h_n[-1]).squeeze(-1)


class ModalityGate(nn.Module):
    """
    Single-modality encoder with SQI-based hard gating.

    When the validity flag is 0, the entire modality output is zeroed out,
    preventing corrupted signals from influencing downstream fusion.
    """

    def __init__(self, in_dim, out_dim=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU())

    def forward(self, x, valid):
        h = self.net(x)
        return h * valid.unsqueeze(-1)


class MultimodalFusion(nn.Module):
    """
    Dual-branch validity-gated multimodal fusion network (Task 1).

    Architecture:
        Physio branch: ECG(3d) + EDA(3d) + RESP(2d) -> ModalityGate x3 -> concat(48d) -> proj(32d)
        Eye branch:    Eye(4d) -> ModalityGate -> proj(16d)
        Fusion:        concat(48d) -> BN -> ReLU -> Dropout(0.5) -> Linear(1)
    """

    def __init__(self):
        super().__init__()
        self.ecg_gate = ModalityGate(3, 16)
        self.eda_gate = ModalityGate(3, 16)
        self.resp_gate = ModalityGate(2, 16)
        self.eye_gate = ModalityGate(4, 16)
        self.physio_proj = nn.Sequential(nn.Linear(48, 32), nn.LayerNorm(32), nn.ReLU())
        self.eye_proj = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(48, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x_ecg, x_eda, x_resp, x_eye, ecg_v, eda_v, resp_v, eye_v):
        h_p = self.physio_proj(torch.cat([
            self.ecg_gate(x_ecg, ecg_v),
            self.eda_gate(x_eda, eda_v),
            self.resp_gate(x_resp, resp_v)
        ], dim=-1))
        h_e = self.eye_proj(self.eye_gate(x_eye, eye_v))
        return self.head(torch.cat([h_p, h_e], dim=-1)).squeeze(-1)


# ======================================================================
# Task 2 Models
# ======================================================================

class SequenceToOneLSTM(nn.Module):
    """2-layer LSTM for session-level SSQ Total Score regression (Task 2).

    Handles variable-length sequences via pack_padded_sequence.
    """

    def __init__(self, input_size=12, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, 2, batch_first=True, dropout=0.3)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1))

    def forward(self, x, lengths):
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        return self.head(h_n[-1]).squeeze(-1)


class ModalityEncoder(nn.Module):
    """Per-timestep modality encoder with validity gating (for sequences)."""

    def __init__(self, input_dim, hidden_dim=16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

    def forward(self, x, valid_mask):
        """
        Args:
            x:          [batch, seq_len, input_dim]
            valid_mask: [batch, seq_len] binary flags
        Returns:
            [batch, seq_len, hidden_dim] with invalid timesteps zeroed out
        """
        h = self.encoder(x)
        mask = valid_mask.unsqueeze(-1)
        return h * mask


class SequenceToOnePredictor(nn.Module):
    """
    Dual-branch validity-gated LSTM fusion for session-level prediction (Task 2).

    Architecture:
        Physio branch: ECG+EDA+RESP -> ModalityEncoder x3 -> concat(48d) -> LSTM(hidden=32)
        Eye branch:    Eye -> ModalityEncoder -> LSTM(hidden=16)
        Fusion:        concat(48d) -> FC -> SSQ score

    Args:
        use_validity_gate: If False, all validity masks default to 1 (ablation mode).
    """

    def __init__(self, use_validity_gate=True):
        super().__init__()
        self.use_validity_gate = use_validity_gate

        self.ecg_encoder = ModalityEncoder(3, 16)
        self.eda_encoder = ModalityEncoder(3, 16)
        self.resp_encoder = ModalityEncoder(2, 16)
        self.eye_encoder = ModalityEncoder(4, 16)

        self.physio_lstm = nn.LSTM(
            input_size=48, hidden_size=32, num_layers=2,
            batch_first=True, dropout=0.3
        )
        self.eye_lstm = nn.LSTM(
            input_size=16, hidden_size=16, num_layers=2,
            batch_first=True, dropout=0.3
        )

        self.fusion_head = nn.Sequential(
            nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x_ecg, x_eda, x_resp, x_eye,
                ecg_valid, eda_valid, resp_valid, eye_valid,
                seq_lengths):
        if not self.use_validity_gate:
            ecg_valid = torch.ones_like(ecg_valid)
            eda_valid = torch.ones_like(eda_valid)
            resp_valid = torch.ones_like(resp_valid)
            eye_valid = torch.ones_like(eye_valid)

        h_ecg = self.ecg_encoder(x_ecg, ecg_valid)
        h_eda = self.eda_encoder(x_eda, eda_valid)
        h_resp = self.resp_encoder(x_resp, resp_valid)
        h_eye = self.eye_encoder(x_eye, eye_valid)

        h_physio = torch.cat([h_ecg, h_eda, h_resp], dim=-1)

        h_physio_packed = nn.utils.rnn.pack_padded_sequence(
            h_physio, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        h_eye_packed = nn.utils.rnn.pack_padded_sequence(
            h_eye, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (h_physio_final, _) = self.physio_lstm(h_physio_packed)
        _, (h_eye_final, _) = self.eye_lstm(h_eye_packed)

        h_physio_out = h_physio_final[-1]
        h_eye_out = h_eye_final[-1]

        h_fused = torch.cat([h_physio_out, h_eye_out], dim=-1)
        output = self.fusion_head(h_fused).squeeze(-1)

        return output
