"""Encode - process - decode graph network (Sanchez-Gonzalez et al. 2020)."""
import torch
import torch.nn as nn

from .graph import N_NODE_FEAT, N_EDGE_FEAT, N_TARGET


def mlp(n_in, hidden, n_out, layer_norm=True):
    layers = [nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out)]
    if layer_norm:
        layers.append(nn.LayerNorm(n_out))
    return nn.Sequential(*layers)


class InteractionLayer(nn.Module):
    """One round of message passing with residual connections."""

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
    """Encode-process-decode, plus a Newton's-third-law acceleration correction.

    The generic decoder can output any acceleration for any node, including a
    coherent push on a whole resting cluster (the observed drift bug). The
    correction below cannot: for every soil-soil edge pair (i,j) it computes one
    scalar strength c_ij from inputs that are *symmetric* under swapping i and j
    (h_i+h_j, |h_i-h_j|, distance — never the raw relative-position vector, which
    flips sign), then applies +c_ij * unit(i-j) to i's acceleration and
    -c_ij * unit(i-j) to j's. The two contributions cancel exactly, so summed
    over any set of particles that only interact with each other (no wall/plate
    edges), the net momentum change from this branch is identically zero — a
    resting interior block cannot be pushed by it, whatever the network learned.
    Wall/plate edges are excluded from this branch (external forces, e.g. the
    floor holding soil up, are not supposed to conserve momentum) and still flow
    through the ordinary message passing above and the generic decoder.
    """

    def __init__(self, hidden=128, layers=6, n_node=N_NODE_FEAT, n_edge=N_EDGE_FEAT, n_out=N_TARGET):
        super().__init__()
        self.node_enc = mlp(n_node, hidden, hidden)
        self.edge_enc = mlp(n_edge, hidden, hidden)
        self.layers = nn.ModuleList(InteractionLayer(hidden) for _ in range(layers))
        self.decoder = mlp(hidden, hidden, n_out, layer_norm=False)
        self.force_mlp = mlp(2 * hidden, hidden, 1, layer_norm=False)
        nn.init.zeros_(self.force_mlp[-1].weight)   # starts as a no-op; training decides how much to use it
        nn.init.zeros_(self.force_mlp[-1].bias)

    def forward(self, x, e, senders, receivers, unit_vec=None, soil_edge=None):
        h = self.node_enc(x)
        e_lat = self.edge_enc(e)
        for layer in self.layers:
            h, e_lat = layer(h, e_lat, senders, receivers)
        out = self.decoder(h)

        if unit_vec is not None:
            sym = torch.cat([h[senders] + h[receivers], (h[senders] - h[receivers]).abs()], dim=1)
            c = self.force_mlp(sym)                                    # (E,1), identical for both directions of a pair
            contrib = c * unit_vec * soil_edge.unsqueeze(1)
            correction = torch.zeros(h.shape[0], 3, device=h.device, dtype=out.dtype).index_add_(0, receivers, contrib)
            out = torch.cat([out[:, :3] + correction, out[:, 3:]], dim=1)
        return out


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
