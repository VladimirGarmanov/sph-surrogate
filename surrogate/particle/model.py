"""Истории через GRU и симметричное объединение соседей -> 16 изменений одной частицы."""
import torch
import torch.nn as nn

from ..model import mlp
from .data import N_EDGE, N_INPUT, N_OUTPUT


class ParticleNet(nn.Module):
    """Без передачи сообщений между соседями; результат не зависит от порядка соседей в списке."""
    def __init__(self, hidden=256, prediction_horizon=1):
        super().__init__()
        if prediction_horizon < 1:
            raise ValueError("prediction_horizon must be positive")
        self.prediction_horizon = prediction_horizon
        self.center_encoder = mlp(N_INPUT, hidden, hidden)
        self.neighbor_encoder = mlp(N_INPUT + N_EDGE, hidden, hidden)
        self.history = nn.GRU(hidden, hidden, batch_first=True)
        self.message = mlp(2 * hidden, hidden, hidden)
        self.decoder = mlp(3 * hidden, hidden, N_OUTPUT * prediction_horizon, layer_norm=False)
        # Изначально после обратной нормализации предсказывается среднее изменение обучающих данных.
        # Строго постоянные нулевые цели сразу предсказываются верно и не дрейфуют.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode_history(self, sequence, encoder):
        """Читаем от старых кадров к новым; скрытое состояние сбрасывается для каждого выбранного окна."""
        _, last_hidden = self.history(encoder(sequence))
        return last_hidden[-1]

    def forward(self, x, neighbors, e, valid):
        # Формы: x — (B,H,18); соседи — (B,K,H,18); рёбра — (B,K,H,7).
        center = self.encode_history(x, self.center_encoder)
        b, k, h, _ = neighbors.shape
        sequences = torch.cat([neighbors, e], dim=-1).reshape(b * k, h, N_INPUT + N_EDGE)
        neighbor = self.encode_history(sequences, self.neighbor_encoder).reshape(b, k, -1)
        query = center[:, None, :].expand_as(neighbor)
        messages = self.message(torch.cat([query, neighbor], dim=-1))
        mask = valid[..., None]
        messages = messages.masked_fill(~mask, 0)
        mean = messages.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        maximum = messages.masked_fill(~mask, -torch.inf).amax(1)
        maximum = torch.where(valid.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        output = self.decoder(torch.cat([center, mean, maximum], dim=-1))
        output = output.reshape(output.shape[0], self.prediction_horizon, N_OUTPUT)
        return output[:, 0] if self.prediction_horizon == 1 else output


def predict(model, batch):
    return model(batch["x"], batch["neighbors"], batch["e"], batch["valid"])
