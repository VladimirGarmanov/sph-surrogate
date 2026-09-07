"""GRU histories + symmetric neighbour pooling -> one particle's 16 deltas."""
import torch
import torch.nn as nn

from ..model import mlp
from .data import N_EDGE, N_INPUT, N_OUTPUT


class ParticleNet(nn.Module):
    """No neighbour-to-neighbour propagation and no dependence on list order."""
    def __init__(self, hidden=256):
        super().__init__()
        self.center_encoder = mlp(N_INPUT, hidden, hidden)
        self.neighbor_encoder = mlp(N_INPUT + N_EDGE, hidden, hidden)
        self.history = nn.GRU(hidden, hidden, batch_first=True)
        self.message = mlp(2 * hidden, hidden, hidden)
        self.decoder = mlp(3 * hidden, hidden, N_OUTPUT, layer_norm=False)
        # Initially predict the training mean delta after denormalization.
        # Exactly constant zero targets start correct, rather than drifting.
        nn.init.zeros_(self.decoder[-1].weight)
        nn.init.zeros_(self.decoder[-1].bias)

    def encode_history(self, sequence, encoder):
        """Read oldest to newest; reset hidden state for each sampled window."""
        _, last_hidden = self.history(encoder(sequence))
        return last_hidden[-1]

    def forward(self, x, neighbors, e, valid):
        # x: (B,H,18); neighbours: (B,K,H,18); edges: (B,K,H,7).
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
        return self.decoder(torch.cat([center, mean, maximum], dim=-1))


def predict(model, batch):
    return model(batch["x"], batch["neighbors"], batch["e"], batch["valid"])
