"""Arquitecturas copiadas de los scripts entregados, con renombrado de clases.

SERGIoT LSTM: experiments/LSTM/lstm_train.py (Model)
iTransformer: experiments/iTransformer/itransformer_train.py
PatchTST: experiments/patchTST/patchtst_train.py
Solo se cambia el nombre de las clases raíz para evitar colisiones.
La adaptación de salidas se hace en models.py, no aquí.
"""
import math
import torch
from torch import nn


# Source: LSTM/lstm_train.py / Model
class SERGIOTLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, horizon):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
        )

        self.linear_1 = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        output, _ = self.lstm(x)
        x = output[:, -1, :]
        x = self.linear_1(x)
        x = self.tanh(x)
        return self.fc(x)


# Source: iTransformer/itransformer_train.py / InvertedEmbedding
class InvertedEmbedding(nn.Module):
    """
    Inverted embedding: each variable (channel), with its full window_size
    temporal window, is projected to a token of dimension d_model. The
    result is N tokens (one per variable), not L tokens (one per time
    step), which is the inversion relative to a Transformer applied to
    time series in the standard way.
    """
    def __init__(self, window_size, d_model, dropout=0.1):
        super().__init__()
        self.value_embedding = nn.Linear(window_size, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, window_size, n_vars) -> (batch, n_vars, window_size)
        x = x.permute(0, 2, 1)
        return self.dropout(self.value_embedding(x))  # (batch, n_vars, d_model)


# Source: iTransformer/itransformer_train.py / EncoderLayer
class EncoderLayer(nn.Module):
    """Standard Transformer encoder layer: self-attention + position-wise FFN + Add&Norm."""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation="gelu"):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU() if activation == "gelu" else nn.ReLU()

    def forward(self, x):
        # Self-attention BETWEEN variables (tokens = variables), no causal mask
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        y = self.dropout(self.activation(self.linear1(x)))
        y = self.dropout(self.linear2(y))
        x = self.norm2(x + y)

        return x, attn_weights


# Source: iTransformer/itransformer_train.py / Encoder
class Encoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x):
        attns = []
        for layer in self.layers:
            x, attn = layer(x)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns


# Source: iTransformer/itransformer_train.py / Modelo
class ITransformerOriginal(nn.Module):
    """
    Full iTransformer: instance normalization -> inverted embedding ->
    encoder (attention between variables) -> linear projection to
    `horizon` steps per variable -> de-normalization.
    Predicts ALL input variables for the whole horizon in a single pass
    (the paper's standard multivariate forecasting).
    """
    def __init__(self, n_vars, window_size, horizon,
                 d_model=128, n_heads=8, e_layers=2, d_ff=128,
                 dropout=0.1, activation="gelu", use_norm=True):
        super().__init__()
        self.use_norm = use_norm

        self.embedding = InvertedEmbedding(window_size, d_model, dropout)
        self.encoder = Encoder(
            [EncoderLayer(d_model, n_heads, d_ff, dropout, activation) for _ in range(e_layers)],
            norm_layer=nn.LayerNorm(d_model)
        )
        self.projector = nn.Linear(d_model, horizon)

    def forward(self, x):
        # x: (batch, window_size, n_vars)
        if self.use_norm:
            means = x.mean(dim=1, keepdim=True).detach()
            x = x - means
            stds = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = x / stds

        tokens = self.embedding(x)          # (batch, n_vars, d_model)
        enc_out, _ = self.encoder(tokens)    # (batch, n_vars, d_model)
        dec_out = self.projector(enc_out)    # (batch, n_vars, horizon)
        dec_out = dec_out.permute(0, 2, 1)   # (batch, horizon, n_vars)

        if self.use_norm:
            dec_out = dec_out * stds.repeat(1, dec_out.size(1), 1) + means.repeat(1, dec_out.size(1), 1)

        return dec_out


# Source: patchTST/patchtst_train.py / PatchEmbedding
class PatchEmbedding(nn.Module):
    def __init__(self, patch_len, stride, d_model, window_size):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.num_patches = (window_size - patch_len) // stride + 1
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x):
        # x: (batch, n_vars, window_size)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: (batch, n_vars, num_patches, patch_len)
        x = self.proj(x)
        # x: (batch, n_vars, num_patches, d_model)
        return x


# Source: patchTST/patchtst_train.py / PositionalEncoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, num_patches, d_model)
        return x + self.pe[:x.size(1)]


# Source: patchTST/patchtst_train.py / Modelo
class PatchTSTOriginal(nn.Module):
    """
    Channel-independent PatchTST, faithful to the original paper:
    - patch embedding and encoder with weights shared across channels.
    - linear head (flatten + linear) ALSO with weights shared across
      channels, applied BEFORE undoing the batch/channel merge. Each
      channel predicts its own future without seeing information from
      the others.
    """
    def __init__(self, n_vars, window_size, horizon, patch_len=4, stride=2,
                 d_model=64, nhead=4, num_encoder_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()

        self.n_vars = n_vars
        self.horizon = horizon
        self.patch_embedding = PatchEmbedding(patch_len, stride, d_model, window_size)
        self.num_patches = self.patch_embedding.num_patches
        self.pos_encoding = PositionalEncoding(d_model, max_len=self.num_patches + 1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        self.dropout = nn.Dropout(dropout)
        # Head shared across channels: receives (num_patches * d_model)
        # and predicts `horizon` steps for ONE channel.
        self.head = nn.Linear(self.num_patches * d_model, horizon)

    def forward(self, x):
        # x: (batch, window_size, n_vars) -> (batch, n_vars, window_size)
        x = x.permute(0, 2, 1)
        batch = x.size(0)

        x = self.patch_embedding(x)  # (batch, n_vars, num_patches, d_model)

        # channel independence: merge batch and n_vars to pass through the
        # same encoder (weights shared across channels)
        x = x.reshape(batch * self.n_vars, self.num_patches, -1)
        x = self.pos_encoding(x)
        x = self.encoder(x)
        x = self.dropout(x)

        # shared head applied channel by channel, WITHOUT merging channels
        # back together (channel independence end to end)
        x = x.reshape(batch * self.n_vars, -1)          # (batch*n_vars, num_patches*d_model)
        out = self.head(x)                               # (batch*n_vars, horizon)
        out = out.reshape(batch, self.n_vars, self.horizon)  # (batch, n_vars, horizon)
        return out

