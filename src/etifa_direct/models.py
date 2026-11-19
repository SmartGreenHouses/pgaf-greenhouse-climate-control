"""Adaptadores multisalida directa de las arquitecturas originales."""
import torch
from torch import nn
from .original_models import SERGIOTLSTM, ITransformerOriginal, PatchTSTOriginal

N_SENSORS = 10

class LSTMStandalone(nn.Module):
    """Equivalente PyTorch de train.py: LSTM(70,tanh) -> Dense.

    Se porta desde TensorFlow/Keras a PyTorch para un protocolo de entrenamiento
    compartido; NO se reproducen los pesos del archivo original.
    """
    def __init__(self, n_inputs, horizon):
        super().__init__()
        self.lstm = nn.LSTM(n_inputs, 70, num_layers=1, batch_first=True)
        self.fc = nn.Linear(70, horizon * N_SENSORS)
        self.horizon = horizon

    def forward(self, x):
        hidden, _ = self.lstm(x)
        return self.fc(hidden[:, -1, :]).reshape(-1, self.horizon, N_SENSORS)

class LSTMSERGIOTMulti(nn.Module):
    """LSTM SERGIoT original; solo amplía la cabeza final a H*10."""
    def __init__(self, n_inputs, horizon):
        super().__init__()
        self.base = SERGIOTLSTM(n_inputs, hidden_size=64, num_layers=1,
                                horizon=horizon * N_SENSORS)
        self.horizon = horizon

    def forward(self, x):
        return self.base(x).reshape(-1, self.horizon, N_SENSORS)

class ITransformerMulti(nn.Module):
    """iTransformer original; selecciona las diez salidas físicas objetivo."""
    def __init__(self, n_inputs, lookback, horizon, use_norm=False):
        super().__init__()
        self.base = ITransformerOriginal(
            n_vars=n_inputs, window_size=lookback, horizon=horizon,
            d_model=128, n_heads=8, e_layers=3, d_ff=128,
            dropout=0.1, activation='gelu', use_norm=use_norm)

    def forward(self, x):
        return self.base(x)[:, :, :N_SENSORS]

class PatchTSTMulti(nn.Module):
    """PatchTST channel-independent original; selecciona diez canales."""
    def __init__(self, n_inputs, lookback, horizon):
        super().__init__()
        self.base = PatchTSTOriginal(
            n_vars=n_inputs, window_size=lookback, horizon=horizon,
            patch_len=24, stride=12, d_model=64, nhead=4,
            num_encoder_layers=3, dim_feedforward=128, dropout=0.1)

    def forward(self, x):
        return self.base(x)[:, :N_SENSORS, :].transpose(1, 2)


class PatchTSTMultiVariable(nn.Module):
    """PatchTST temporal con atención entre canales para ETIFA.

    Entrada:  [B, L, 19]
    Salida:   [B, H, 10]

    La atención entre canales permite que las predicciones de
    los sensores dependan de sensores, actuadores y tiempo.
    No se utilizan variables futuras.
    """

    def __init__(self, n_inputs, lookback, horizon):
        super().__init__()
        self.n_inputs = n_inputs
        self.horizon = horizon

        self.base = PatchTSTOriginal(
            n_vars=n_inputs,
            window_size=lookback,
            horizon=horizon,
            patch_len=24,
            stride=12,
            d_model=64,
            nhead=4,
            num_encoder_layers=3,
            dim_feedforward=128,
            dropout=0.1,
        )

        # PatchTST-MV reutiliza únicamente el encoder temporal.
        # La head channel-independent original no participa en forward().
        self.base.head = nn.Identity()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(64)

        self.fusion_head = nn.Linear(
            self.base.num_patches * 64 + 64,
            horizon,
        )

    def forward(self, x):
        batch = x.size(0)

        # [B,L,C] -> [B,C,L]
        x = x.permute(0, 2, 1)

        # Representación temporal independiente por canal
        x = self.base.patch_embedding(x)
        x = x.reshape(
            batch * self.n_inputs,
            self.base.num_patches,
            64,
        )

        x = self.base.pos_encoding(x)
        x = self.base.encoder(x)
        x = self.base.dropout(x)

        # [B,C,P,D]
        x = x.reshape(
            batch,
            self.n_inputs,
            self.base.num_patches,
            64,
        )

        # Resumen temporal por variable
        channel_tokens = x.mean(dim=2)

        # Atención entre sensores, actuadores y tiempo
        attended, _ = self.cross_attention(
            channel_tokens,
            channel_tokens,
            channel_tokens,
            need_weights=False,
        )
        context = self.cross_norm(channel_tokens + attended)

        # Preservar los parches temporales completos
        temporal = x.reshape(batch, self.n_inputs, -1)

        fused = torch.cat([temporal, context], dim=-1)

        # Solo se predicen los diez sensores
        predictions = self.fusion_head(fused[:, :N_SENSORS, :])

        return predictions.transpose(1, 2)

class Persistence(nn.Module):
    def __init__(self, horizon):
        super().__init__()
        self.horizon = horizon

    def forward(self, x):
        return x[:, -1:, :N_SENSORS].expand(-1, self.horizon, -1)


def build_model(name, lookback, horizon, n_inputs=19):
    if name == 'lstm_original':
        return LSTMStandalone(n_inputs, horizon)
    if name == 'lstm_sergiot':
        return LSTMSERGIOTMulti(n_inputs, horizon)
    if name == 'itransformer':
        return ITransformerMulti(n_inputs, lookback, horizon, use_norm=False)
    if name == 'itransformer_norm_true':
        return ITransformerMulti(n_inputs, lookback, horizon, use_norm=True)
    if name == 'itransformer_norm_false':
        return ITransformerMulti(n_inputs, lookback, horizon, use_norm=False)
    if name == 'patchtst':
        return PatchTSTMulti(n_inputs, lookback, horizon)
    if name == 'patchtst_mv':
        return PatchTSTMultiVariable(n_inputs, lookback, horizon)
    if name == 'persistence':
        return Persistence(horizon)
    raise ValueError(f'Modelo desconocido: {name}')
