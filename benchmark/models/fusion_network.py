"""SQI-Aware multimodal fusion networks for VR cybersickness prediction.

Both models use Signal Quality Index (SQI) validity flags as hard-gating masks,
zeroing out features from modalities with unreliable signal at each time step.
"""

import torch
import torch.nn as nn


class ModalityGate(nn.Module):
    """Single-modality encoder with SQI-based hard gating.

    Applies a linear projection followed by LayerNorm and ReLU, then masks
    the output to zero for time steps where the modality validity flag is 0.

    Args:
        in_dim: Input feature dimension for this modality.
        out_dim: Output embedding dimension.
    """

    def __init__(self, in_dim: int, out_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Modality features, shape (batch, in_dim).
            valid: Binary validity flag, shape (batch,).
        Returns:
            Gated embedding, shape (batch, out_dim).
        """
        return self.net(x) * valid.unsqueeze(-1)


class MultimodalFusion(nn.Module):
    """Point-wise SQI-Aware Multimodal Fusion for minute-level prediction (Task 1).

    Fuses ECG, EDA, RESP, and eye-tracking modalities using per-sample validity
    flags as hard-gating masks. Physiological modalities are projected jointly;
    eye-tracking is projected separately before final fusion.

    Architecture:
        ECG (3-d) + EDA (3-d) + RESP (2-d) → ModalityGate × 3 → physio_proj (48→32)
        Eye (4-d)                            → ModalityGate     → eye_proj   (16→16)
        Concat (48-d) → BatchNorm → Dropout(0.5) → Linear → score
    """

    def __init__(self):
        super().__init__()
        self.ecg_gate  = ModalityGate(3, 16)
        self.eda_gate  = ModalityGate(3, 16)
        self.resp_gate = ModalityGate(2, 16)
        self.eye_gate  = ModalityGate(4, 16)
        self.physio_proj = nn.Sequential(nn.Linear(48, 32), nn.LayerNorm(32), nn.ReLU())
        self.eye_proj    = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(48, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.5), nn.Linear(32, 1)
        )

    def forward(self, x_ecg, x_eda, x_resp, x_eye, ecg_v, eda_v, resp_v, eye_v):
        h_p = self.physio_proj(torch.cat([
            self.ecg_gate(x_ecg, ecg_v),
            self.eda_gate(x_eda, eda_v),
            self.resp_gate(x_resp, resp_v),
        ], dim=-1))
        h_e = self.eye_proj(self.eye_gate(x_eye, eye_v))
        return self.head(torch.cat([h_p, h_e], dim=-1)).squeeze(-1)


# ---------------------------------------------------------------------------
# Task 2: sequence-level SQI-Aware fusion (per-timestep gating + LSTM)
# ---------------------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """Per-timestep modality encoder with SQI-based hard gating for sequences.

    Args:
        input_dim: Feature dimension of this modality.
        hidden_dim: Output embedding dimension per time step.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Shape (batch, seq_len, input_dim).
            valid_mask: Binary mask, shape (batch, seq_len).
        Returns:
            Gated embeddings, shape (batch, seq_len, hidden_dim).
        """
        return self.encoder(x) * valid_mask.unsqueeze(-1)


class SequenceToOnePredictor(nn.Module):
    """Sequence-level SQI-Aware Multimodal Fusion for session-level prediction (Task 2).

    Processes complete 10–16 minute physiological sequences to predict the
    session-level SSQ Total Score. Per-timestep validity flags gate each
    modality independently before dual-branch LSTM encoding.

    Architecture:
        ECG+EDA+RESP → ModalityEncoder × 3 → concat (48-d) → LSTM(hidden=32)
        Eye           → ModalityEncoder     →               → LSTM(hidden=16)
        Concat (48-d) → FC → SSQ score

    Args:
        use_validity_gate: If False, all validity masks are set to 1 (ablation).
    """

    def __init__(self, use_validity_gate: bool = True):
        super().__init__()
        self.use_validity_gate = use_validity_gate
        self.ecg_encoder  = ModalityEncoder(3, 16)
        self.eda_encoder  = ModalityEncoder(3, 16)
        self.resp_encoder = ModalityEncoder(2, 16)
        self.eye_encoder  = ModalityEncoder(4, 16)
        self.physio_lstm  = nn.LSTM(48, 32, 2, batch_first=True, dropout=0.3)
        self.eye_lstm     = nn.LSTM(16, 16, 2, batch_first=True, dropout=0.3)
        self.fusion_head  = nn.Sequential(
            nn.Linear(48, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 1)
        )

    def forward(self, x_ecg, x_eda, x_resp, x_eye,
                ecg_valid, eda_valid, resp_valid, eye_valid,
                seq_lengths: torch.Tensor) -> torch.Tensor:
        if not self.use_validity_gate:
            ecg_valid  = torch.ones_like(ecg_valid)
            eda_valid  = torch.ones_like(eda_valid)
            resp_valid = torch.ones_like(resp_valid)
            eye_valid  = torch.ones_like(eye_valid)

        h_physio = torch.cat([
            self.ecg_encoder(x_ecg, ecg_valid),
            self.eda_encoder(x_eda, eda_valid),
            self.resp_encoder(x_resp, resp_valid),
        ], dim=-1)
        h_eye = self.eye_encoder(x_eye, eye_valid)

        packed_p = nn.utils.rnn.pack_padded_sequence(
            h_physio, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_e = nn.utils.rnn.pack_padded_sequence(
            h_eye, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (hp, _) = self.physio_lstm(packed_p)
        _, (he, _) = self.eye_lstm(packed_e)

        return self.fusion_head(torch.cat([hp[-1], he[-1]], dim=-1)).squeeze(-1)
