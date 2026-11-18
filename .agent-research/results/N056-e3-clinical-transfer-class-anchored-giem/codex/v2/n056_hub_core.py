"""Verbatim N054 v3 hub architecture and objective helpers."""
import math, torch
from torch import nn
from torch.nn import functional as F

def mlp(i, o, hidden=512, depth=2):
    layers = []
    for _ in range(depth): layers += [nn.Linear(i, hidden), nn.GELU(), nn.LayerNorm(hidden)]; i = hidden
    return nn.Sequential(*layers, nn.Linear(i, o))

class SigLIP(nn.Module):
    def __init__(self):
        super().__init__(); self.t = nn.Parameter(torch.tensor(math.log(10.))); self.b = nn.Parameter(torch.tensor(-10.))
    def forward(self, a, b, ignore=None):
        a, b = F.normalize(a, dim=-1), F.normalize(b, dim=-1)
        logits = a @ b.T * self.t.exp().clamp(max=100) + self.b
        labels = 2 * torch.eye(len(a), device=a.device) - 1
        loss = -F.logsigmoid(labels * logits)
        if ignore is not None: loss = loss.masked_fill(ignore, 0)
        return loss.sum() / len(a)

def near_duplicate(y, m, tau=.25):
    # Surface-labelled rows: identical observed surface sets and nearby values.
    # Autofl-only rows also contain a real scalar readout and need the same rule.
    surf, sm = y[:, :12], m[:, :12]
    same = (sm[:, None] == sm[None]).all(-1) & (sm.sum(1)[:, None] > 0)
    diff = ((surf[:, None] - surf[None]).abs() * sm[:, None]).amax(-1)
    af = m[:, 12] & ~sm.any(-1)
    ignore = (same & (diff < tau)) | (af[:,None] & af[None] & ((y[:,12,None]-y[None,:,12]).abs() < tau))
    ignore.fill_diagonal_(False)
    return ignore

def mse_mask(pred, y, m):
    # Equal target weighting (prevents large single-marker slides dominating).
    num = ((pred - y).square() * m).sum(0); den = m.sum(0)
    return (num / den.clamp(min=1))[den > 0].mean() if (den > 0).any() else pred.sum() * 0

class Spaces(nn.Module):
    def __init__(self, dim=256):
        super().__init__(); self.P = mlp(1024, dim); self.EM = mlp(26, dim)
        self.H = mlp(dim, 13); self.sig = SigLIP(); self.dim = dim
    def lf(self, x): return F.normalize(self.P(x), dim=-1)
    def marker(self, y, m): return F.normalize(self.EM(torch.cat([y * m, m.float()], -1)), dim=-1)
    def decode(self, z): return self.H(z * math.sqrt(self.dim))
    def forward(self, x): return self.decode(self.lf(x))
