"""
Temporal Fusion Transformer (TFT) for Stock Price Forecasting
Simplified but powerful implementation from scratch in PyTorch.

Paper: https://arxiv.org/abs/1912.09363
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# 1. Gated Residual Network (GRN)
# Core building block of TFT
# ─────────────────────────────────────────────
class GatedResidualNetwork(nn.Module):
    """
    GRN applies a non-linear transformation with a gating mechanism.
    Gate controls how much of the transformed signal passes through.
    This helps the model ignore irrelevant features automatically.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1      = nn.Linear(input_dim, hidden_dim)
        self.fc2      = nn.Linear(hidden_dim, output_dim)
        self.gate     = nn.Linear(hidden_dim, output_dim)
        self.skip     = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout  = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        residual = self.skip(x)
        h        = F.elu(self.fc1(x))
        h        = self.dropout(h)
        out      = self.fc2(h)
        gate     = torch.sigmoid(self.gate(h))
        out      = gate * out
        return self.layer_norm(out + residual)


# ─────────────────────────────────────────────
# 2. Multi-Head Attention (Interpretable)
# ─────────────────────────────────────────────
class InterpretableMultiHeadAttention(nn.Module):
    """
    Modified multi-head attention that shares values across heads.
    This makes attention weights interpretable — you can see which
    past time steps the model is attending to when making predictions.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads
        self.W_q      = nn.Linear(d_model, d_model)
        self.W_k      = nn.Linear(d_model, d_model)
        self.W_v      = nn.Linear(d_model, self.d_k)  # shared across heads
        self.W_o      = nn.Linear(d_model, d_model)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, T, _ = q.shape

        Q = self.W_q(q).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(k).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(v).unsqueeze(1).expand(-1, self.n_heads, -1, -1)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.W_o(context), attn_weights.mean(dim=1)


# ─────────────────────────────────────────────
# 3. Positional Encoding
# ─────────────────────────────────────────────
class PositionalEncoding(nn.Module):
    """Adds position information to embeddings."""
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


# ─────────────────────────────────────────────
# 4. Full TFT Model
# ─────────────────────────────────────────────
class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-step time series forecasting.

    Architecture:
        Input features (OHLCV + technical indicators)
          └─ Feature embedding (Linear projection to d_model)
              └─ Positional Encoding
                  └─ LSTM encoder (captures local temporal patterns)
                      └─ GRN (gates irrelevant features)
                          └─ Multi-Head Self-Attention (long-range dependencies)
                              └─ GRN + LayerNorm
                                  └─ Linear → forecast_horizon outputs

    Args:
        input_dim       : number of input features (OHLCV + indicators)
        d_model         : internal embedding dimension
        n_heads         : attention heads
        n_lstm_layers   : LSTM encoder layers
        forecast_horizon: how many days ahead to predict
        dropout         : regularisation
    """
    def __init__(self, input_dim=16, d_model=64, n_heads=4,
                 n_lstm_layers=2, forecast_horizon=5, dropout=0.1):
        super().__init__()
        self.forecast_horizon = forecast_horizon
        self.d_model          = d_model

        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size  = d_model,
            hidden_size = d_model,
            num_layers  = n_lstm_layers,
            batch_first = True,
            dropout     = dropout if n_lstm_layers > 1 else 0,
        )

        # GRN after LSTM
        self.grn1 = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)

        # Interpretable multi-head attention
        self.attention  = InterpretableMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_norm  = nn.LayerNorm(d_model)

        # GRN after attention
        self.grn2 = GatedResidualNetwork(d_model, d_model * 2, d_model, dropout)

        # Output head — predicts forecast_horizon steps ahead
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, input_dim) — historical features
        Returns:
            predictions: (batch, forecast_horizon) — future prices
            attn_weights: (batch, seq_len, seq_len) — interpretable attention
        """
        # Project input to d_model dimensions
        x = self.input_proj(x)
        x = self.pos_enc(x)

        # LSTM: captures short-term temporal patterns
        lstm_out, _ = self.lstm(x)

        # GRN: gates irrelevant signals
        grn_out = self.grn1(lstm_out)

        # Self-attention: captures long-range dependencies
        attn_out, attn_weights = self.attention(grn_out, grn_out, grn_out)
        attn_out = self.attn_norm(attn_out + grn_out)

        # Second GRN
        grn_out2 = self.grn2(attn_out)

        # Use the last time step's representation for forecasting
        last_hidden = grn_out2[:, -1, :]

        # Predict forecast_horizon days ahead
        predictions = self.output_head(last_hidden)

        return predictions, attn_weights


def get_model(input_dim=16, forecast_horizon=5):
    return TemporalFusionTransformer(
        input_dim=input_dim,
        d_model=64,
        n_heads=4,
        n_lstm_layers=2,
        forecast_horizon=forecast_horizon,
        dropout=0.1
    )


if __name__ == "__main__":
    model = get_model(input_dim=16, forecast_horizon=5)
    x     = torch.randn(8, 60, 16)   # batch=8, seq=60 days, features=16
    preds, attn = model(x)
    print(f"Input  : {x.shape}")
    print(f"Output : {preds.shape}")   # (8, 5) — 5-day forecast
    print(f"Attn   : {attn.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Params : {total:,}")
