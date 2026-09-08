"""Графовая сеть: кодирование, обработка, декодирование (Sanchez-Gonzalez и др., 2020)."""
import torch
import torch.nn as nn

from .graph import N_NODE_FEAT, N_EDGE_FEAT, N_TARGET


def mlp(n_in, hidden, n_out, layer_norm=True):
    layers = [nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out)]
    if layer_norm:
        layers.append(nn.LayerNorm(n_out))
    return nn.Sequential(*layers)


class InteractionLayer(nn.Module):
    """Один раунд передачи сообщений с остаточными связями."""

    def __init__(self, hidden):
        super().__init__()
        self.edge_mlp = mlp(3 * hidden, hidden, hidden)
        self.node_mlp = mlp(2 * hidden, hidden, hidden)

    def forward(self, h, e, senders, receivers):
        e_new = e + self.edge_mlp(torch.cat([e, h[senders], h[receivers]], dim=1))
        agg = torch.zeros_like(h).index_add_(0, receivers, e_new)
        h_new = h + self.node_mlp(torch.cat([h, agg], dim=1))
        return h_new, e_new


class GNS(nn.Module):
    def __init__(self, hidden=128, layers=6, n_node=N_NODE_FEAT, n_edge=N_EDGE_FEAT, n_out=N_TARGET):
        super().__init__()
        self.node_enc = mlp(n_node, hidden, hidden)
        self.edge_enc = mlp(n_edge, hidden, hidden)
        self.layers = nn.ModuleList(InteractionLayer(hidden) for _ in range(layers))
        self.decoder = mlp(hidden, hidden, n_out, layer_norm=False)

    def forward(self, x, e, senders, receivers):
        h = self.node_enc(x)
        e = self.edge_enc(e)
        for layer in self.layers:
            h, e = layer(h, e, senders, receivers)
        return self.decoder(h)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
