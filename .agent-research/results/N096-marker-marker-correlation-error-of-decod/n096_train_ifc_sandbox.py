#!/usr/bin/env python3
"""Donor-held-out S-BIAD452 joint decoding. See README.md for estimands and caveats."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

MARKERS = ['CD15', 'Siglec8', 'CD14', 'CD19', 'CD3', 'CD45', 'CD4', 'CD56', 'CD8']
MI = {m: i for i, m in enumerate(MARKERS)}
POOL_MEMBERS = [MI[m] for m in ('CD3', 'CD19', 'CD56')]
VERSION = 1


class EvaluationInterrupted(Exception):
    pass


def donor_ids(samples):
    out = []
    for sample in samples:
        match = re.fullmatch(r'sample([012])(?:_(?:early|late))?', str(sample))
        if match is None:
            raise ValueError(f'Unrecognised donor/sample: {sample!r}')
        out.append(int(match[1]))
    return np.asarray(out)


def donor_split(samples, test_donor):
    """Fixed cyclic rule: test d, validation (d+1)%3, train (d+2)%3."""
    d = donor_ids(samples)
    split = tuple(np.flatnonzero(d == k) for k in ((test_donor + 2) % 3,
                                                  (test_donor + 1) % 3, test_donor))
    if any(len(x) == 0 for x in split):
        raise ValueError('All three donors must be present (including in smoke packs).')
    return split


def reveal_masks(rows, regime, seed):
    """Stateless per packed-row seed; masks do not change with epoch, subset, or resume."""
    rows = np.asarray(rows, dtype=np.uint64)
    # SplitMix64: independent streams for marker and panel assignment.
    def uniform(stream):
        with np.errstate(over='ignore'):
            z = rows + np.uint64(seed) + np.uint64(stream)
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
            z ^= z >> np.uint64(31)
        return (z >> np.uint64(11)).astype(np.float64) / 2**53
    mask = np.zeros((len(rows), 9), bool)
    pool = np.zeros((len(rows), 2), bool)
    if regime == 'full':
        mask[:] = True
    elif regime in ('single', 'single_pooled'):
        mask[np.arange(len(rows)), np.minimum((uniform(0x12345) * 9).astype(int), 8)] = True
        if regime == 'single_pooled':
            pooled = uniform(0x98765) < 0.1
            mask[pooled] = False
            pool[pooled] = True
    else:
        raise ValueError(regime)
    return mask, pool


def masked_mse(pred, target, mask):
    # Select BEFORE subtraction: NaNs or arbitrary hidden labels cannot enter loss.
    if not mask.any():
        return pred.sum() * 0
    return (pred[mask] - target[mask]).square().mean()


def robust_scale(x):
    return max(float(1.4826 * np.median(np.abs(x - np.median(x)))), 1.0)


def fit_pool_calibration(raw, y, single_mask, pooled_mask):
    """Fit only revealed singles/pooled observations, never hidden component values."""
    visible = pooled_mask[:, 0]
    if visible.sum() < 2:
        raise ValueError('single_pooled needs >=2 pooled TRAIN cells; increase smoke --max-train.')
    sums = raw[:, POOL_MEMBERS].sum(1)
    c = robust_scale(sums[visible])
    # P1 = asinh(linear intensity sum / c); P2 = linear CD45, as requested.
    p = np.stack([np.arcsinh(sums / c), raw[:, MI['CD45']]], 1)
    mean, std = p[visible].mean(0), np.maximum(p[visible].std(0), 1e-6)
    slope, bias = np.ones(9), np.zeros(9)
    for j in range(9):
        obs = single_mask[:, j]
        if obs.sum() < 2:
            raise ValueError(f'No inverse-map calibration support for {MARKERS[j]}; increase training rows.')
        x, t = y[obs, j], np.arcsinh(raw[obs, j] / c)
        a = np.mean((x - x.mean()) * (t - t.mean())) / max(float(x.var()), 1e-8)
        slope[j] = np.clip(a, 1e-3, 20)
        bias[j] = np.clip(t.mean() - slope[j] * x.mean(), -15, 15)
    return dict(c=c, mean=mean.tolist(), std=std.tolist(), slope=slope.tolist(), bias=bias.tolist(),
                n_pooled=int(visible.sum()), n_single=single_mask.sum(0).tolist())


def pooled_targets(raw, calibration):
    p = np.stack([np.arcsinh(raw[:, POOL_MEMBERS].sum(1) / calibration['c']),
                  raw[:, MI['CD45']]], 1)
    return ((p - calibration['mean']) / calibration['std']).astype(np.float32)


class MonotoneInverse(nn.Module):
    """y -> c*sinh(positive_slope*y+bias), jointly learned using revealed data only."""
    def __init__(self, calibration):
        super().__init__()
        slopes = torch.tensor(calibration['slope'], dtype=torch.float32)
        self.log_slope = nn.Parameter(torch.log(torch.expm1(slopes)))
        self.bias = nn.Parameter(torch.tensor(calibration['bias'], dtype=torch.float32))
        self.register_buffer('c', torch.tensor(calibration['c'], dtype=torch.float32))
        self.register_buffer('mean', torch.tensor(calibration['mean'], dtype=torch.float32))
        self.register_buffer('std', torch.tensor(calibration['std'], dtype=torch.float32))

    def transformed(self, y):
        return F.softplus(self.log_slope) * y + self.bias

    def forward(self, y):
        # Saturation protects sinh under runaway predictions; map remains nondecreasing.
        raw_over_c = torch.sinh(self.transformed(y.float()).clamp(-20, 20))
        p = torch.stack([torch.asinh(raw_over_c[:, POOL_MEMBERS].sum(1)),
                         self.c * raw_over_c[:, MI['CD45']]], 1)
        return (p - self.mean) / self.std


class MarkerPrior(nn.Module):
    """Decode a latent into the sandbox target coordinates. All fitted tensors are frozen."""
    def __init__(self, kind, dim):
        super().__init__()
        self.dim = dim
        self.register_buffer('center', torch.zeros(9))
        self.register_buffer('scale', torch.ones(9))
        self.decoder = (nn.Linear(dim, 9, bias=False) if kind == 'linear' else
                        nn.Sequential(nn.Linear(dim, 64), nn.GELU(), nn.Linear(64, 9)))

    def forward(self, latent):
        return self.center + self.scale * self.decoder(latent)


def marker_prior(kind, dim, val_y, seed, device):
    """val_y contains all nine marker measurements in the validation fold's target units.

    On resume only build the module; the checkpoint restores the fitted decoder and scaling.
    Separate RNG streams keep image encoder initialization independent of prior fitting.
    """
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        prior = MarkerPrior(kind, dim).to(device)
        if val_y is not None:
            if len(val_y) <= dim:
                raise ValueError('Need more validation profiles than prior dimensions.')
            y = torch.as_tensor(val_y, dtype=torch.float32, device=device)
            prior.center.copy_(y.mean(0))
            prior.scale.copy_(y.std(0, unbiased=False).clamp_min(1e-6))
            standardized = (y - prior.center) / prior.scale
            if kind == 'linear':
                _, vectors = torch.linalg.eigh(standardized.T @ standardized)
                with torch.no_grad():
                    prior.decoder.weight.copy_(vectors[:, -dim:].flip(1))
            else:
                encoder = nn.Sequential(nn.Linear(9, 64), nn.GELU(), nn.Linear(64, dim)).to(device)
                ae = nn.Sequential(encoder, prior.decoder)
                optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
                generator = torch.Generator(device=device).manual_seed(seed)
                for _ in range(30):
                    for rows in torch.randperm(len(y), generator=generator, device=device).split(512):
                        batch = standardized[rows]
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.mse_loss(ae(batch), batch)
                        loss.backward()
                        optimizer.step()
        return prior.requires_grad_(False).eval()


class SandboxModel(nn.Module):
    def __init__(self, cov_rank=0, pooled_mode='none', calibration=None, tiny=False, prior=None):
        super().__init__()
        if tiny:
            self.encoder = nn.Sequential(nn.Conv2d(3, 8, 3, stride=4, padding=1), nn.ReLU(),
                                         nn.AdaptiveAvgPool2d(1), nn.Flatten())
            nf = 8
        else:
            import timm
            self.encoder = timm.create_model('resnet50', pretrained=False, in_chans=3, num_classes=0)
            nf = self.encoder.num_features
        self.mean = nn.Linear(nf, prior.dim if prior is not None else 9)
        self.prior = prior
        self.rank = cov_rank
        self.cov = nn.Linear(nf, 9 * (1 + cov_rank)) if cov_rank else None
        self.pool_head = nn.Linear(nf, 2) if pooled_mode == 'head' else None
        self.inverse = MonotoneInverse(calibration) if pooled_mode == 'inverse' else None

    def forward(self, x):
        f = self.encoder(x)
        mu = self.mean(f)
        if self.prior is not None:
            mu = self.prior(mu)
        cov = None
        if self.cov is not None:
            z = self.cov(f).float()
            cov = (F.softplus(z[:, :9]) + 1e-4, z[:, 9:].reshape(-1, 9, self.rank))
        pool = self.pool_head(f) if self.pool_head is not None else None
        if self.inverse is not None:
            pool = self.inverse(mu)
        return mu, pool, cov, f


def gaussian_nll(mu, y, cov, reduction='mean'):
    diag, factor = cov
    covariance = torch.diag_embed(diag.float()) + factor.float() @ factor.float().transpose(-1, -2)
    chol = torch.linalg.cholesky(covariance)
    error = (y.float() - mu.float()).unsqueeze(-1)
    mahal = (error * torch.cholesky_solve(error, chol)).sum((-1, -2))
    nll = 0.5 * (mahal + 2 * chol.diagonal(dim1=-2, dim2=-1).log().sum(-1) + 9 * math.log(2 * math.pi))
    return nll.mean() if reduction == 'mean' else nll


class Cells(torch.utils.data.Dataset):
    def __init__(self, pack, rows):
        self.path, self.rows = str(Path(pack) / 'lf.npy'), np.asarray(rows)
        self.images = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self.images is None:
            self.images = np.load(self.path, mmap_mode='r')
        row = int(self.rows[index])
        return torch.from_numpy(np.array(self.images[row], dtype=np.float32)), row


def augment(x):
    """Independent D4 rotations/reflections and zero-filled +-4px shifts per cell."""
    n, _, h, w = x.shape
    k = torch.randint(4, (n,), device=x.device)
    out = torch.empty_like(x)
    for rotation in range(4):
        select = k == rotation
        out[select] = torch.rot90(x[select], rotation, (-2, -1))
    flip = torch.rand(n, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    shifts = torch.randint(-4, 5, (n, 2), device=x.device)
    yy = torch.arange(h, device=x.device)[None, :, None] - shifts[:, 0, None, None]
    xx = torch.arange(w, device=x.device)[None, None, :] - shifts[:, 1, None, None]
    valid = (yy >= 0) & (yy < h) & (xx >= 0) & (xx < w)
    shifted = out.permute(0, 2, 3, 1)[torch.arange(n, device=x.device)[:, None, None],
                                        yy.clamp(0, h - 1), xx.clamp(0, w - 1)]
    return (shifted * valid[..., None]).permute(0, 3, 1, 2).contiguous()


def pearson(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) < 3:
        return None
    a, b = a - a.mean(), b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.clip(a @ b / denom, -1, 1)) if denom > 1e-12 else None


def average_defined(values):
    good = [v for v in values if v is not None]
    return float(np.mean(good)) if good else None


def marker_correlations(y):
    return [[pearson(y[:, i], y[:, j]) for j in range(9)] for i in range(9)]


def fit_gates(train_y):
    """Evaluation-only empirical thresholds, never supplied to model/loss."""
    return dict(positive=np.quantile(train_y, .75, axis=0).tolist(),
                cd45_hi=float(np.quantile(train_y[:, MI['CD45']], .25)),
                cd15_lo=float(np.quantile(train_y[:, MI['CD15']], .5)))


def gate_masks(y, gates):
    pos = y > np.asarray(gates['positive'])
    cd = lambda m: pos[:, MI[m]]
    wbc = y[:, MI['CD45']] > gates['cd45_hi']
    lymph = wbc & (y[:, MI['CD15']] <= gates['cd15_lo']) & ~cd('CD14')
    gran = wbc & cd('CD15')
    mono = wbc & ~cd('CD15') & cd('CD14')
    return dict(lymphocytes=lymph, granulocytes=gran, monocytes=mono,
                T_cells=lymph & cd('CD3'), B_cells=lymph & ~cd('CD3') & cd('CD19'),
                NK_cells=lymph & ~cd('CD3') & ~cd('CD19') & cd('CD56'),
                CD4_T=lymph & cd('CD3') & cd('CD4'), CD8_T=lymph & cd('CD3') & cd('CD8'),
                eosinophils=gran & cd('Siglec8'), neutrophils=gran & ~cd('Siglec8'),
                CD3_CD19_DP=cd('CD3') & cd('CD19'))


def coarse_populations(y, gates):
    masks = gate_masks(y, gates)
    assigned = np.zeros(len(y), bool)
    out = {}
    for name in ('lymphocytes', 'granulocytes', 'monocytes'):
        out[name] = masks[name]
        assigned |= out[name]
    out['other'] = ~assigned
    return out


def correlation_fidelity(y, p):
    t, q = marker_correlations(y), marker_correlations(p)
    valid = [(i, j) for i in range(9) for j in range(9)
             if t[i][j] is not None and q[i][j] is not None]
    # Undefined correlations are not replaced with zero or silently scored as correct.
    frob = math.sqrt(sum((q[i][j] - t[i][j]) ** 2 for i, j in valid)) if valid else None
    return dict(n=int(len(y)), frobenius=frob, valid_entries=len(valid), total_entries=81,
                true=t, predicted=q)


def fscore(truth, prediction):
    tp = int((truth & prediction).sum())
    fp = int((~truth & prediction).sum())
    fn = int((truth & ~prediction).sum())
    denom = 2 * tp + fp + fn
    return dict(f1=2 * tp / denom if denom else None, tp=tp, fp=fp, fn=fn,
                true_count=int(truth.sum()), predicted_count=int(prediction.sum()))


def compute_metrics(y, p, gates):
    y, p = np.asarray(y, np.float64), np.asarray(p, np.float64)
    if y.shape != p.shape or y.ndim != 2 or y.shape[1] != 9 or not np.isfinite([y, p]).all():
        raise ValueError('Metrics require finite matching (N,9) truth and predictions.')
    r = {m: pearson(y[:, j], p[:, j]) for j, m in enumerate(MARKERS)}
    rmse = np.sqrt(np.mean((y - p) ** 2, axis=1))
    norms = np.linalg.norm(y, axis=1) * np.linalg.norm(p, axis=1)
    cos = np.divide((y * p).sum(1), norms, out=np.full(len(y), np.nan), where=norms > 1e-12)
    populations = coarse_populations(y, gates)
    coexpression = {'overall': correlation_fidelity(y, p)}
    within = {}
    means_y, means_p = [], []
    for name, mask in populations.items():
        coexpression[name] = correlation_fidelity(y[mask], p[mask])
        rr = {m: pearson(y[mask, j], p[mask, j]) for j, m in enumerate(MARKERS)}
        within[name] = dict(n=int(mask.sum()), per_marker_r=rr, macro_r=average_defined(rr.values()))
        if mask.any():
            means_y.append(y[mask].mean(0))
            means_p.append(p[mask].mean(0))
    cluster_r = {m: pearson(np.asarray(means_y)[:, j], np.asarray(means_p)[:, j])
                 for j, m in enumerate(MARKERS)}
    pair_delta = {}
    for i in range(9):
        for j in range(i + 1, 9):
            tr, pr = pearson(y[:, i], y[:, j]), pearson(p[:, i], p[:, j])
            pair_delta[f'{MARKERS[i]}--{MARKERS[j]}'] = dict(true_r=tr, predicted_r=pr,
                predicted_minus_true=None if tr is None or pr is None else pr - tr)
    key_pairs = {}
    for pop in ('lymphocytes', 'granulocytes'):
        mask = populations[pop]
        key_pairs[pop] = {}
        for a, b in [('CD3', 'CD19'), ('CD3', 'CD56'), ('CD4', 'CD8'), ('CD14', 'CD15')]:
            key_pairs[pop][f'{a}--{b}'] = dict(n=int(mask.sum()),
                true_r=pearson(y[mask, MI[a]], y[mask, MI[b]]),
                predicted_r=pearson(p[mask, MI[a]], p[mask, MI[b]]))
    true_gates, pred_gates = gate_masks(y, gates), gate_masks(p, gates)
    gating = {name: fscore(mask, pred_gates[name]) for name, mask in true_gates.items()}
    return dict(per_marker_r=r, macro_r=average_defined(r.values()),
                vector_rmse_mean=float(rmse.mean()), vector_rmse_median=float(np.median(rmse)),
                vector_cosine_mean=float(np.nanmean(cos)) if np.isfinite(cos).any() else None,
                vector_cosine_defined=int(np.isfinite(cos).sum()),
                coexpression=coexpression, within_population=within,
                population_means=dict(n_populations=len(means_y), per_marker_r=cluster_r,
                                      macro_r=average_defined(cluster_r.values())),
                pairwise_correlation_bias=pair_delta, key_pairs=key_pairs, gating=gating,
                double_positive_CD3_CD19=dict(true_rate=float(true_gates['CD3_CD19_DP'].mean()),
                                              predicted_rate=float(pred_gates['CD3_CD19_DP'].mean())))


def knn_impute(train_features, test_features, train_y, mask, k=50, query_chunk=256,
               reference_chunk=4096, device='cpu', should_stop=None):
    """Exact Euclidean KNN; both axes chunked, identical revealed banks searched once."""
    if k < 1:
        raise ValueError('K must be positive.')
    train_features = torch.as_tensor(train_features, dtype=torch.float32, device=device)
    test_features = torch.as_tensor(test_features, dtype=torch.float32, device=device)
    result = np.empty((len(test_features), 9), np.float32)
    counts, groups = [], {}
    for j in range(9):
        selected = np.flatnonzero(mask[:, j])
        if not len(selected):
            raise ValueError(f'No revealed training neighbours for {MARKERS[j]}')
        counts.append(dict(marker=MARKERS[j], bank_size=len(selected), effective_k=min(k, len(selected))))
        groups.setdefault(mask[:, j].tobytes(), []).append(j)
    for columns in groups.values():
        selected = np.flatnonzero(mask[:, columns[0]])
        kk = min(k, len(selected))
        bank = train_features[torch.as_tensor(selected, device=device)]
        values = np.asarray(train_y)[np.ix_(selected, columns)]
        for start in range(0, len(test_features), query_chunk):
            q = test_features[start:start + query_chunk]
            distances = torch.full((len(q), kk), float('inf'), device=device)
            indices = torch.zeros((len(q), kk), dtype=torch.long, device=device)
            for ref in range(0, len(bank), reference_chunk):
                if should_stop is not None and should_stop():
                    raise EvaluationInterrupted
                b = bank[ref:ref + reference_chunk]
                dist = (q.square().sum(1)[:, None] + b.square().sum(1)[None] - 2 * q @ b.T).clamp_min_(0)
                ids = torch.arange(ref, ref + len(b), device=device)[None].expand(len(q), -1)
                combined = torch.cat([distances, dist], 1)
                distances, pos = combined.topk(kk, dim=1, largest=False, sorted=True)
                indices = torch.cat([indices, ids], 1).gather(1, pos)
            result[start:start + len(q), columns] = values[indices.cpu().numpy()].mean(1)
    return result, counts


def write_json(path, value):
    temp = Path(str(path) + '.tmp')
    with temp.open('w') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    os.replace(temp, path)


def save_checkpoint(path, payload):
    temp = str(path) + '.tmp'
    torch.save(payload, temp)
    os.replace(temp, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def epoch_batches(rows, batch_size, seed, epoch):
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch])).permutation(rows)
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    # BatchNorm cannot train on a singleton. Keep that cell by merging the last batch.
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2] = np.concatenate([batches[-2], batches[-1]])
        batches.pop()
    return batches


def loader(pack, rows, args, batches=None):
    # Separate generator: constructing/skipping loaders must not consume augmentation RNG.
    generator = torch.Generator().manual_seed(args.seed + 101)
    kw = dict(num_workers=args.workers, pin_memory=args.device == 'cuda', generator=generator)
    if batches is None:
        return torch.utils.data.DataLoader(Cells(pack, rows), batch_size=args.bs, **kw)
    # Cells rows are flattened, and batch_sampler indexes that same flattened sequence.
    sizes = [len(b) for b in batches]
    offsets = np.cumsum([0] + sizes)
    sampler = [list(range(offsets[i], offsets[i + 1])) for i in range(len(sizes))]
    return torch.utils.data.DataLoader(Cells(pack, np.concatenate(batches)), batch_sampler=sampler, **kw)


def autocast(args):
    return torch.autocast('cuda', dtype=torch.bfloat16) if args.device == 'cuda' else nullcontext()


def batch_loss(model, output, ids, targets, masks, pools, pool_masks, raw, calibration, args, corr=True):
    mu, pp, cov, _ = output
    y = targets[ids].to(args.device)
    mask = masks[ids].to(args.device)
    loss = gaussian_nll(mu, y, cov) if cov is not None else masked_mse(mu.float(), y, mask)
    if pp is not None:
        loss = loss + args.pool_weight * masked_mse(pp.float(), pools[ids].to(args.device),
                                                   pool_masks[ids].to(args.device))
    if corr and getattr(model, 'corr_target', None) is not None:   # N096 red-team #3: checkpoint selection uses the N080 loss only
        z = mu.float() - mu.float().mean(0)
        z = z / (z.square().mean(0).sqrt() + 1e-6)
        r = (z.T @ z) / len(z)
        iu = torch.triu_indices(9, 9, 1, device=r.device)
        loss = loss + args.corr_match * (r[iu[0], iu[1]] - model.corr_target[iu[0], iu[1]]).square().mean()
    if model.inverse is not None and args.map_weight:
        # These are the SAME observed single stains, in raw/c coordinates; no extra labels.
        observed_raw = raw[ids].to(args.device)
        map_target = torch.asinh(observed_raw / calibration['c'])
        loss = loss + args.map_weight * masked_mse(model.inverse.transformed(y), map_target, mask)
    return loss


@torch.no_grad()
def validation_loss(model, pack, rows, scale, tensors, calibration, args):
    model.eval()
    total, count = 0., 0
    for x, ids in loader(pack, rows, args):
        with autocast(args):
            output = model(x.to(args.device) / scale)
            loss = batch_loss(model, output, ids, *tensors, calibration, args, corr=False)
        # Skip cells without any observations (there are none for the supported regimes).
        total += float(loss) * len(x)
        count += len(x)
    return total / count


@torch.no_grad()
def predict(model, pack, rows, scale, args, should_stop=None):
    model.eval()
    means, features, diagonals, factors = [], [], [], []
    for x, _ in loader(pack, rows, args):
        if should_stop is not None and should_stop():
            raise EvaluationInterrupted
        with autocast(args):
            mu, _, cov, embedding = model(x.to(args.device) / scale)
        means.append(mu.float().cpu().numpy())
        features.append(embedding.float().cpu().numpy())
        if cov is not None:
            diagonals.append(cov[0].cpu().numpy())
            factors.append(cov[1].cpu().numpy())
    return (np.concatenate(means), np.concatenate(features),
            (np.concatenate(diagonals), np.concatenate(factors)) if diagonals else None)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--pack', default='/hnvme/workspace/v181bb10-bsccm/external/sbiad452_pack')
    ap.add_argument('--out', required=True)
    ap.add_argument('--regime', choices=['full', 'single', 'single_pooled'], default='single')
    ap.add_argument('--test-donor', type=int, choices=[0, 1, 2], required=True)
    ap.add_argument('--pooled-mode', choices=['inverse', 'head'], default='inverse')
    ap.add_argument('--marker-prior', choices=['none', 'linear', 'ae'], default='none')
    ap.add_argument('--prior-dim', type=int, choices=range(1, 10), default=4)
    ap.add_argument('--cov-rank', type=int, choices=[0, 2], default=0)
    # N096 (user 11-27, marker-marker correlation error): penalise the batch correlation matrix of the predicted means against a
    # FIXED target correlation matrix: 'val' = the validation donor's joint labels (the same external joint data the N087 marker prior
    # uses; allowed in every regime), 'train' = the training donor's labels (only in --regime full, where all are revealed).
    ap.add_argument('--corr-match', type=float, default=0., help='weight of the 36-pair correlation-matching loss (0 = off)')
    ap.add_argument('--corr-source', choices=['val', 'train'], default='val')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--lr', type=float, default=5e-4)
    ap.add_argument('--pool-weight', type=float, default=1.)
    ap.add_argument('--map-weight', type=float, default=.1)
    ap.add_argument('--knn-impute', type=int, default=50, metavar='K', help='0 disables the baseline')
    ap.add_argument('--knn-query-chunk', type=int, default=256)
    ap.add_argument('--knn-reference-chunk', type=int, default=4096)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    ap.add_argument('--threads', type=int, default=4)
    ap.add_argument('--checkpoint-every', type=int, default=200)
    ap.add_argument('--max-train', type=int, default=0)
    ap.add_argument('--max-val', type=int, default=0)
    ap.add_argument('--max-test', type=int, default=0)
    ap.add_argument('--smoke-tiny-model', action='store_true', help='testing ONLY; replaces resnet50')
    ap.add_argument('--stop-after-steps', type=int, default=0, help='testing: interrupt after N new steps')
    a = ap.parse_args(argv)
    if a.corr_source == 'train' and a.regime != 'full':
        ap.error('--corr-source train needs --regime full (hidden training labels must not reach the loss)')
    if not np.isfinite(a.corr_match) or a.corr_match < 0:
        ap.error('--corr-match must be finite and nonnegative')
    if a.cov_rank and a.regime != 'full':
        ap.error('--cov-rank 2 requires --regime full')
    for name in ('epochs', 'threads', 'checkpoint_every', 'knn_query_chunk', 'knn_reference_chunk'):
        if getattr(a, name) < 1:
            ap.error(f'{name} must be positive')
    if a.bs < 2:
        ap.error('--bs must be >=2')
    for name in ('seed', 'workers', 'knn_impute', 'max_train', 'max_val', 'max_test', 'stop_after_steps'):
        if getattr(a, name) < 0:
            ap.error(f'{name} must be nonnegative')
    if not np.isfinite(a.lr) or a.lr <= 0:
        ap.error('--lr must be finite and positive')
    for name in ('pool_weight', 'map_weight'):
        if not np.isfinite(getattr(a, name)) or getattr(a, name) < 0:
            ap.error(f'{name} must be finite and nonnegative')
    return a


def main(argv=None):
    args = parse_args(argv)
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable: use --device cpu for tiny smoke tests.')
    pack, out = Path(args.pack).resolve(), Path(args.out).resolve()
    if out == pack or pack in out.parents:
        raise ValueError('--out must not be inside the read-only pack.')
    out.mkdir(parents=True, exist_ok=True)
    with np.load(pack / 'labels.npz', allow_pickle=False) as data:
        labels = {k: data[k] for k in data.files}
    y, raw = labels['Y'].astype(np.float32), labels['I_raw'].astype(np.float32)
    if list(labels['names'].astype(str)) != MARKERS or y.shape != raw.shape or y.shape[1:] != (9,):
        raise ValueError('Expected canonical nine-marker schema.')
    if not np.isfinite(y).all() or not np.isfinite(raw).all():
        raise ValueError('All packed cells must have nine finite measurements.')
    images = np.load(pack / 'lf.npy', mmap_mode='r')
    if images.shape != (len(y), 3, 128, 128) or images.dtype != np.float16:
        raise ValueError('Expected float16 lf.npy (N,3,128,128).')
    chan_scale = np.asarray(labels['chan_scale'], dtype=np.float32)
    if chan_scale.shape != (3,) or not np.isfinite(chan_scale).all() or (chan_scale <= 0).any():
        raise ValueError('chan_scale must contain three positive finite values.')
    scale = torch.tensor(chan_scale, device=args.device)[None, :, None, None]
    tr, va, te = donor_split(labels['sample'], args.test_donor)
    def subset(rows, limit, stream):
        return np.sort(np.random.default_rng(args.seed + stream).choice(rows, min(limit, len(rows)), replace=False)) if limit else rows
    tr, va, te = subset(tr, args.max_train, 1), subset(va, args.max_val, 2), subset(te, args.max_test, 3)
    if len(tr) < 2:
        raise ValueError('Need >=2 training cells.')
    masks, pool_masks = reveal_masks(np.arange(len(y)), args.regime, args.seed)
    # N080 integration (Claude): fold-specific preprocessing fitted on the TRAINING donor's REVEALED entries only (the pack fits them on
    # --stat-donors; hidden labels must not influence anything, cf. test_hidden_labels_and_test_truth_cannot_change_trained_model):
    # c_m = max(1.4826 MAD of revealed training I_raw, 1) -> asinh(I/c_m); per-sample-run z: training AND validation runs from revealed
    # entries (validation selects the checkpoint), test-donor runs from their own truth (metrics only); chan_scale from <= 5,000 training cells.
    _tr_set = np.zeros(len(y), bool); _tr_set[tr] = True; _yt = np.empty_like(raw); c_tr = np.ones(9, np.float32)
    for j in range(9):
        v = raw[_tr_set & masks[:, j], j]
        if len(v) >= 2: c_tr[j] = max(1.4826 * float(np.median(np.abs(v - np.median(v)))), 1.0)
    _yt = np.arcsinh(raw / c_tr); samp = labels['sample'].astype(str); y = np.zeros_like(_yt)
    _is_test = donor_ids(labels['sample']) == args.test_donor
    for sv in np.unique(samp):
        k = samp == sv
        for j in range(9):
            ref = _yt[k, j] if _is_test[k].all() else _yt[k & masks[:, j], j]   # test-donor runs: own truth (metrics only); training/validation runs: revealed entries only
            mu, sd = (float(ref.mean()), float(ref.std()) + 1e-6) if len(ref) >= 2 else (0.0, 1.0)
            y[k, j] = (_yt[k, j] - mu) / sd
    _rs = np.sort(np.random.default_rng(0).choice(tr, min(len(tr), 5000), replace=False))
    _cen = np.asarray(images[_rs, :, 32:96, 32:96], dtype=np.float32)
    chan_scale = np.array([max(float(np.std(_cen[:, c][_cen[:, c] != 0])), 1e-3) for c in range(3)], np.float32); del _cen
    scale = torch.tensor(chan_scale, device=args.device)[None, :, None, None]
    print('N080 fold stats (training donor, revealed only): cofactor', np.round(c_tr, 1).tolist(), 'chan_scale', np.round(chan_scale, 4).tolist(), flush=True)
    calibration = fit_pool_calibration(raw[tr], y[tr], masks[tr], pool_masks[tr]) if args.regime == 'single_pooled' else None
    pools = pooled_targets(raw, calibration) if calibration else np.zeros((len(y), 2), np.float32)
    # Seal supervision: hidden training labels are replaced by zero before constructing tensors.
    visible_y = np.where(masks, y, 0).astype(np.float32)
    visible_raw = np.where(masks, raw, 0).astype(np.float32)
    pools = np.where(pool_masks, pools, 0).astype(np.float32)
    tensors = tuple(torch.from_numpy(x) for x in (visible_y, masks, pools, pool_masks, visible_raw))
    mode = args.pooled_mode if args.regime == 'single_pooled' else 'none'
    prior = None
    reported_args = vars(args).copy()
    if args.marker_prior == 'none':
        # Preserve archived control configs, results, checkpoints and RNG stream.
        del reported_args['marker_prior'], reported_args['prior_dim']
    else:
        val_profiles = None if (out / 'ckpt.pt').exists() else y[va]
        prior = marker_prior(args.marker_prior, args.prior_dim, val_profiles, args.seed, args.device)
    model = SandboxModel(args.cov_rank, mode, calibration, args.smoke_tiny_model, prior).to(args.device)
    model.corr_target = None
    if args.corr_match > 0:
        src = va if args.corr_source == 'val' else tr   # 'train' only in --regime full (parse_args), where every label is revealed
        model.corr_target = torch.as_tensor(np.corrcoef(y[src].T), dtype=torch.float32, device=args.device)
        write_json(out / 'corr_target.json', dict(source=args.corr_source, n=int(len(src)), R=np.corrcoef(y[src].T).round(4).tolist()))
    else:
        del reported_args['corr_match'], reported_args['corr_source']   # archived configs unchanged
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.05)
    steps_per_epoch = len(epoch_batches(tr, args.bs, args.seed, 0))
    total_steps = args.epochs * steps_per_epoch
    # Avoid OneCycle's zero-length warmup at exactly ten tiny smoke steps.
    pct_start = max(.1, 2 / total_steps) if total_steps >= 3 else .1
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=total_steps, pct_start=pct_start)
    excluded = {'out', 'workers', 'threads', 'checkpoint_every', 'stop_after_steps',
                'knn_impute', 'knn_query_chunk', 'knn_reference_chunk'}
    cfg = {k: v for k, v in reported_args.items() if k not in excluded}
    cfg['pack'] = str(pack)
    cfg['version'] = VERSION
    cfg['labels_sha256'] = hashlib.sha256((pack / 'labels.npz').read_bytes()).hexdigest()
    stat = (pack / 'lf.npy').stat()
    cfg['lf_stat'] = [stat.st_size, stat.st_mtime_ns]
    ckpt_path = out / 'ckpt.pt'
    epoch, batch, step, loss_sum, loss_count = 0, 0, 0, 0., 0
    best_loss, best_state, history = math.inf, None, []
    resumed = False
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if ckpt['cfg'] != cfg or any(not np.array_equal(ckpt[k], v) for k, v in [('tr', tr), ('va', va), ('te', te)]):
            raise ValueError('Refusing resume: recipe, pack fingerprint, or donor rows changed.')
        model.load_state_dict(ckpt['model'])
        opt.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        epoch, batch, step = ckpt['epoch'], ckpt['batch'], ckpt['step']
        loss_sum, loss_count = ckpt['loss_sum'], ckpt['loss_count']
        best_loss, best_state, history = ckpt['best_loss'], ckpt['best_state'], ckpt['history']
        restore_rng(ckpt['rng'])
        resumed = True
        print(f'Resumed epoch={epoch} batch={batch} step={step}', flush=True)
    setup = dict(cfg=cfg, split=dict(train_donor=(args.test_donor + 2) % 3,
                                   validation_donor=(args.test_donor + 1) % 3, test_donor=args.test_donor,
                                   n_train=len(tr), n_validation=len(va), n_test=len(te)),
                 n_revealed_singles=masks[tr].sum(0).tolist(), n_pooled_cells=int(pool_masks[tr, 0].sum()),
                 pool_calibration=calibration, architecture='tiny_smoke' if args.smoke_tiny_model else 'resnet50',
                 target_normalization='Pack Y: per-sample z scores using all sample cells (retrospective calibration).',
                 pooled_normalization='Fit training donor revealed pooled cells; freeze for validation/test.')
    write_json(out / 'setup.json', setup)
    stop = {'signal': None}
    def request_stop(signum, frame):
        stop['signal'] = signal.Signals(signum).name
    old_handlers = {s: signal.signal(s, request_stop) for s in (signal.SIGUSR1, signal.SIGTERM)}
    start_step, start_time = step, time.monotonic()
    def checkpoint():
        save_checkpoint(ckpt_path, dict(cfg=cfg, model=model.state_dict(), optimizer=opt.state_dict(),
            scheduler=scheduler.state_dict(), epoch=epoch, batch=batch, step=step,
            loss_sum=loss_sum, loss_count=loss_count, best_loss=best_loss, best_state=best_state,
            history=history, tr=tr, va=va, te=te, rng=rng_state()))
    try:
        while epoch < args.epochs:
            batches = epoch_batches(tr, args.bs, args.seed, epoch)
            model.train()
            if batch < len(batches):
                for x, ids in loader(pack, tr, args, batches[batch:]):
                    opt.zero_grad(set_to_none=True)
                    with autocast(args):
                        output = model(augment(x.to(args.device) / scale))
                        loss = batch_loss(model, output, ids, *tensors, calibration, args)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Nonfinite loss; last good checkpoint is preserved.')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                    opt.step()
                    scheduler.step()
                    step += 1
                    batch += 1
                    loss_sum += float(loss.detach()) * len(x)
                    loss_count += len(x)
                    if args.stop_after_steps and step - start_step >= args.stop_after_steps:
                        stop['signal'] = 'test_stop_after_steps'
                    if step % args.checkpoint_every == 0 or stop['signal']:
                        checkpoint()
                    if stop['signal']:
                        write_json(out / 'result.json', dict(status='interrupted', reason=stop['signal'],
                                   epoch=epoch, batch=batch, step=step, resumed=resumed))
                        return 0
            val = validation_loss(model, pack, va, scale, tensors, calibration, args)
            if not np.isfinite(val):
                raise FloatingPointError('Nonfinite validation loss.')
            if val < best_loss:
                best_loss = val
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            history.append(dict(epoch=epoch + 1, step=step, train_loss=loss_sum / loss_count, validation_loss=val))
            print(json.dumps(history[-1]), flush=True)
            epoch, batch, loss_sum, loss_count = epoch + 1, 0, 0., 0
            checkpoint()
            if stop['signal']:
                write_json(out / 'result.json', dict(status='interrupted', reason=stop['signal'], epoch=epoch, step=step))
                return 0
        # Only the fixed validation donor selects the checkpoint. Test donor is scored once here.
        model.load_state_dict(best_state)
        should_stop = lambda: stop['signal'] is not None
        prediction, test_features, cov = predict(model, pack, te, scale, args, should_stop)
        gates = fit_gates(y[tr])  # evaluation only; full train truth never feeds any loss or representation
        metrics = compute_metrics(y[te], prediction, gates)
        payload = dict(P=prediction, Y=y[te], rows=te, names=np.asarray(MARKERS), sample=labels['sample'][te],
                       vector_rmse=np.sqrt(np.mean((prediction - y[te]) ** 2, axis=1)))
        norms = np.linalg.norm(prediction, axis=1) * np.linalg.norm(y[te], axis=1)
        payload['vector_cosine'] = np.divide((prediction * y[te]).sum(1), norms,
            out=np.full(len(te), np.nan, np.float32), where=norms > 1e-12)
        for key in ('part', 'object_number'):
            if key in labels:
                payload[key] = labels[key][te]
        if cov is not None:
            payload.update(cov_diag=cov[0], cov_factor=cov[1])
            metrics['gaussian_nll'] = float(gaussian_nll(torch.from_numpy(prediction), torch.from_numpy(y[te]),
                                          tuple(torch.from_numpy(a) for a in cov)))
        if args.knn_impute:
            _, train_features, _ = predict(model, pack, tr, scale, args, should_stop)
            knn, counts = knn_impute(train_features, test_features, visible_y[tr], masks[tr], args.knn_impute,
                                    args.knn_query_chunk, args.knn_reference_chunk, device=args.device, should_stop=should_stop)
            metrics['knn'] = compute_metrics(y[te], knn, gates)
            metrics['knn']['neighbours'] = counts
            payload['P_knn'] = knn
        with (out / 'eval_preds.npz.tmp').open('wb') as handle:
            np.savez_compressed(handle, **payload)
        os.replace(out / 'eval_preds.npz.tmp', out / 'eval_preds.npz')
        write_json(out / 'metrics.json', dict(test_donor=args.test_donor, n_test=len(te),
                   names=MARKERS, evaluation_thresholds=gates, model=metrics))
        write_json(out / 'result.json', dict(status='complete', args=reported_args, setup=setup,
                   final=metrics, history=history, steps=step, resumed=resumed,
                   selected_epoch=min(history, key=lambda h: h['validation_loss'])['epoch'],
                   invocation_seconds=time.monotonic() - start_time))
        print(json.dumps(dict(status='complete', macro_r=metrics['macro_r'], n_test=len(te))), flush=True)
        return 0
    except EvaluationInterrupted:
        # Training has already been checkpointed; preserve its last state (not the selected best).
        write_json(out / 'result.json', dict(status='interrupted', reason=stop['signal'],
                   epoch=epoch, step=step, phase='evaluation', resumed=resumed))
        return 0
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    raise SystemExit(main())
