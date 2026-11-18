"""Source-only N060--N062 heads. forward always returns 13 label-unit predictions."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

DIM = ('single:CD123', 'single:HLA-DR', 'single:CD19', 'single:CD56')
PANELS = {'grp:CD123/HLA-DR/CD14': ('CD123', 'HLA-DR', 'CD14'),
          'grp:CD3/CD19/CD56': ('CD3', 'CD19', 'CD56'),
          'grp:CD45': ('CD45',), 'grp:CD16': ('CD16',)}


def masked_mean(value, mask, weight, target_weight):
    """Legacy per-target masked/weighted mean, then weighted present-target mean."""
    mw = mask * weight
    per = (value * mw).sum(0) / mw.sum(0).clamp_min(1e-6)
    active = mask.any(0) * target_weight
    return (per * active).sum() / active.sum().clamp_min(1e-6)


class HLGauss(nn.Module):
    def __init__(self, nf, y, mask, bins=64):
        super().__init__()
        edges = []
        for j in range(y.shape[1]):
            v = y[mask[:, j], j]
            lo, hi = np.quantile(v, [.001, .999]) if len(v) else (-1., 1.)
            # Add 10% of the quantile span on EACH side. Unobserved targets
            # use a placeholder support and remain fully masked in the loss.
            span = max(float(hi - lo), 1e-3)
            edges.append(np.linspace(lo - .1 * span, hi + .1 * span, bins + 1))
        self.register_buffer('edges', torch.tensor(np.array(edges), dtype=torch.float32))
        self.register_buffer('centres', (self.edges[:, :-1] + self.edges[:, 1:]) / 2)
        self.linear = nn.Linear(nf, y.shape[1] * bins)
        self.k, self.bins = y.shape[1], bins

    def logits(self, h):
        return self.linear(h).reshape(-1, self.k, self.bins)

    def forward(self, h):
        return (self.logits(h).softmax(-1) * self.centres).sum(-1)

    def target_probs(self, y):
        # Gaussian bin integrals, truncated and normalized over support. Clip
        # out-of-support means for stable tails; expectation MSE keeps raw y.
        mean = y.clamp(self.edges[:, 0], self.edges[:, -1])
        sigma = .75 * (self.edges[:, 1] - self.edges[:, 0])
        cdf = .5 * (1 + torch.erf((self.edges[None] - mean[..., None]) /
                                 (math.sqrt(2) * sigma[None, :, None])))
        prob = (cdf[..., 1:] - cdf[..., :-1]).clamp_min(0)
        return prob / prob.sum(-1, keepdim=True).clamp_min(1e-12)

    def cell_loss(self, h, y):
        return -(self.target_probs(y) * self.logits(h).log_softmax(-1)).sum(-1)


class TwoPart(nn.Module):
    def __init__(self, nf, names, y, mask, pos_q):
        super().__init__()
        self.base = nn.Linear(nf, len(names))
        self.register_buffer('indices', torch.tensor([names.index(n) for n in DIM]))
        self.parts = nn.Linear(nf, len(DIM) * 3)
        thresholds, weights, active, means = [], [], [], []
        for j in self.indices.tolist():
            v = y[mask[:, j], j]
            thr = float(np.quantile(v, pos_q)) if len(v) else 0.
            pos = v > thr
            np_, nn_ = int(pos.sum()), int((~pos).sum())
            active.append(np_ > 0 and nn_ > 0)
            thresholds.append(thr)
            weights.append([len(v) / (2 * max(nn_, 1)), len(v) / (2 * max(np_, 1))])
            means.append([float(v[~pos].mean()) if nn_ else 0.,
                          float(v[pos].mean()) if np_ else 0.])
        self.register_buffer('thresholds', torch.tensor(thresholds))
        self.register_buffer('class_weight', torch.tensor(weights))
        self.register_buffer('gate_active', torch.tensor(active))
        with torch.no_grad():
            bias = self.parts.bias.reshape(-1, 3)
            bias[:, 0] = 0
            bias[:, 1:] = torch.tensor(means)  # negative, positive amplitude

    def components(self, h):
        return self.parts(h).reshape(-1, len(DIM), 3)

    def forward(self, h):
        p = self.base(h)
        v = self.components(h)
        gate = v[..., 0].sigmoid()
        p[:, self.indices] = gate * v[..., 2] + (1 - gate) * v[..., 1]
        return p

    def gate_loss(self, h, y, mask, weight, target_weight):
        yy = y[:, self.indices] > self.thresholds
        cw = torch.where(yy, self.class_weight[:, 1], self.class_weight[:, 0])
        value = F.binary_cross_entropy_with_logits(self.components(h)[..., 0], yy.float(), reduction='none') * cw
        # This is a per-target addition to MSE: retain the SAME present-target
        # denominator, with zero BCE for ordinary/unsupported target heads.
        dense = torch.zeros_like(y)
        dense[:, self.indices] = value * self.gate_active
        return masked_mean(dense, mask, weight, target_weight)


class PanelHead(nn.Module):
    """8 factors, monotone assay links, and a free autofl head.

    W learns positive weights on panel edges, with exact zeros outside them.
    The pooled decoder is a small monotone affine + tanh link: single/pooled
    labels have different asinh/z coordinates and are never equated directly.
    """
    def __init__(self, nf, names):
        super().__init__()
        singles = [n for n in names if n.startswith('single:')]
        if len(names) != 13 or len(singles) != 8 or names[-1] != 'autofl':
            raise ValueError('panel-head requires the 13-target BSCCM schema')
        groups = [n for n in names if n.startswith('grp:')]
        design = np.array([[n.split(':', 1)[1] in PANELS[g] for n in singles] for g in groups], np.float32)
        self.register_buffer('design', torch.tensor(design))
        self.z = nn.Linear(nf, 8)
        self.autofl = nn.Linear(nf, 1)
        one = math.log(math.expm1(1.))
        self.single_slope = nn.Parameter(torch.full((8,), one))
        self.single_bias = nn.Parameter(torch.zeros(8))
        self.raw_w = nn.Parameter(torch.full((4, 8), one))
        self.pool_slope = nn.Parameter(torch.full((4,), one))
        self.pool_curve = nn.Parameter(torch.full((4,), -4.))
        self.pool_bias = nn.Parameter(torch.zeros(4))

    def forward(self, h):
        z = self.z(h)
        single = F.softplus(self.single_slope) * z + self.single_bias
        u = F.linear(z, self.design * F.softplus(self.raw_w))
        pooled = F.softplus(self.pool_slope) * u + F.softplus(self.pool_curve) * u.tanh() + self.pool_bias
        return torch.cat([single, pooled, self.autofl(h)], 1)


class TrainAdapters(nn.Module):
    """Tables contain only selected training slides/FOVs; called in loss only."""
    def __init__(self, fov, slide, k, fov_re=False, panel=False):
        super().__init__()
        uf, fi = np.unique(np.stack([slide, fov], 1), axis=0, return_inverse=True)
        us, si = np.unique(slide, return_inverse=True)
        self.register_buffer('source_fovs', torch.tensor(uf))
        self.register_buffer('source_slides', torch.tensor(us.astype(np.int64)))
        self.fov_re, self.panel = fov_re, panel
        if fov_re:
            self.offset = nn.Parameter(torch.zeros(len(uf), k))
            self.delta = nn.Parameter(torch.zeros(len(us), 3, k))
            self.shift = nn.Parameter(torch.zeros(len(us), 3, k))
        if panel:
            self.pool_delta = nn.Parameter(torch.zeros(len(us), 4))
            self.pool_shift = nn.Parameter(torch.zeros(len(us), 4))
        self.fi, self.si = fi, si  # selected source-row order

    def forward(self, p, l3, fi, si):
        if (fi < 0).any() or (si < 0).any():
            raise ValueError('training adapters received a non-training row')
        q = p
        if self.panel:
            q = torch.cat([q[:, :8], (1 + self.pool_delta[si]) * q[:, 8:12] + self.pool_shift[si], q[:, 12:]], 1)
        if self.fov_re:
            cls = l3.detach().argmax(1)  # predicted class, never true gates
            q = (1 + self.delta[si, cls]) * q + self.shift[si, cls] + self.offset[fi]
        return q

    def penalty(self, fov_weight, shrink, target_weight):
        # Whole-table shrinkage, with identity represented by zero delta.
        def avg(v, tw):
            per_target = v.square().reshape(-1, v.shape[-1]).mean(0)
            return (per_target * tw).sum() / tw.sum().clamp_min(1e-6)
        loss = target_weight.sum() * 0
        if self.fov_re:
            loss = loss + fov_weight * (avg(self.offset, target_weight) + shrink *
                                       (avg(self.delta, target_weight) + avg(self.shift, target_weight)))
        if self.panel:
            loss = loss + shrink * (avg(self.pool_delta, target_weight[8:12]) + avg(self.pool_shift, target_weight[8:12]))
        return loss


def pair_difference_loss(p, y, mask, fov, target_weight, rows=None):
    """All unordered distinct-row same-FOV pairs, averaged per observed target."""
    same = (fov[:, None] == fov[None, :]).triu(diagonal=1)
    if rows is not None:
        same &= rows[:, None] != rows[None, :]
    pm = same[..., None] & mask[:, None, :] & mask[None, :, :]
    e = p - y
    error = (e[:, None, :] - e[None, :, :]).square()
    count = pm.sum((0, 1))
    per = (error * pm).sum((0, 1)) / count.clamp_min(1)
    active = (count > 0) * target_weight
    return (per * active).sum() / active.sum().clamp_min(1e-6)


class FOVBatchSampler:
    """bs/k cells per each of k source FOVs; floor(N/bs) batches per epoch.

    Count-proportional FOV draws without replacement within a batch. Small
    FOVs sample cells with replacement; pair loss excludes duplicate rows.
    This intentionally changes the training sampling recipe.
    """
    def __init__(self, fov, slide, bs, k, seed):
        _, ids = np.unique(np.stack([slide, fov], 1), axis=0, return_inverse=True)
        order = np.argsort(ids, kind='stable')
        self.groups = np.split(order, np.flatnonzero(np.diff(ids[order])) + 1)
        if bs % k or bs // k < 2 or k > len(self.groups):
            raise ValueError('fov-batch requires bs divisible by k, bs/k >= 2, and >= k source FOVs')
        self.prob = np.array([len(g) for g in self.groups], dtype=float)
        self.prob /= self.prob.sum()
        self.bs, self.k, self.n = bs, k, len(ids) // bs
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return self.n

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.n):
            fs = rng.choice(len(self.groups), self.k, replace=False, p=self.prob)
            rows = [rng.choice(self.groups[f], self.bs // self.k,
                               replace=len(self.groups[f]) < self.bs // self.k) for f in fs]
            yield rng.permutation(np.concatenate(rows)).tolist()
