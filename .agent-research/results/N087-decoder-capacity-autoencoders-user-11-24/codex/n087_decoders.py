"""Fit frozen-feature decoders on one draw and measure Pearson r on the other."""
import argparse
from pathlib import Path
import re
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_goal_arms import aggregate_goal, common11, target_scores


RANKS = (1, 2, 4, 8, 12, 'full')   # Claude integration (red-team m4): k >= 13 equals full rank for a 13-output linear map
# Coefficients in the objective mean squared error + lambda * ||w||^2.
LAMBDAS = (1e-4, 1e-3, 1e-2, 1e-1, 1., 10., 100., 1e3, 1e4, 1e5)   # Claude integration (red-team m2): finer, wider grid; choice logged
RECORDED = {'b10': .6867, 'b01': .6592, 'GOAL': .6730}


def inner_split(slides, fovs, seed):
    _, groups = np.unique(np.column_stack((slides, fovs)), axis=0, return_inverse=True)
    splitter = GroupShuffleSplit(n_splits=1, test_size=.1, random_state=seed)
    return next(splitter.split(groups, groups=groups))


def scaler(x):
    sd, mean = torch.std_mean(x, dim=0, correction=0)
    return mean, torch.where(sd > 0, sd, 1.)


def ridge_path(x, y, queries, lambdas):
    """Solve all penalties from one centered Gram matrix; intercept is unpenalized."""
    x, y = x.double(), y.double()
    mu, ybar = x.mean(0), y.mean()
    x -= mu
    gram = (x.T @ x) / len(x)
    eig, basis = torch.linalg.eigh(gram)
    rhs = basis.T @ (x.T @ (y - ybar) / len(x))
    weights = basis @ (rhs[:, None] / (eig.clamp_min(0)[:, None] + lambdas))
    return ((queries.double() - mu) @ weights + ybar).float()


def ridge(raw, x, queries, y, mask, fit, val):
    """Scale the selection fold afresh; refit the selected penalty on all source rows."""
    output = x.new_full((len(queries), y.shape[1]), float('nan'))
    alphas = torch.tensor(LAMBDAS, device=x.device, dtype=torch.float64)
    mu, sd = scaler(raw[fit])
    for j in range(y.shape[1]):
        a, b = fit[mask[fit, j]], val[mask[val, j]]
        if not len(a) or not len(b):
            continue
        pv = ridge_path((raw[a] - mu) / sd, y[a, j], (raw[b] - mu) / sd, alphas)
        best = (pv - y[b, j, None]).square().mean(0).argmin()
        print(f'ridge target {j}: lambda {float(alphas[best]):g}', flush=True)
        use = mask[:, j]
        output[:, j] = ridge_path(x[use], y[use, j], queries, alphas[best].reshape(1))[:, 0]
    return output


def network(width, outputs, decoder, k):
    if decoder == 'mlp':
        return nn.Sequential(nn.Linear(width, 512), nn.GELU(), nn.Dropout(.1),
                             nn.Linear(512, 512), nn.GELU(), nn.Dropout(.1),
                             nn.Linear(512, outputs))
    if k == 'full':
        return nn.Linear(width, outputs)
    return nn.Sequential(nn.Linear(width, k), nn.Linear(k, outputs))


def train_network(x, q, y, mask, decoder, k, *, seed, epochs, bs, lr):
    torch.manual_seed(seed)
    model = network(x.shape[1], y.shape[1], decoder, k).to(x.device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.)
    # Discard rows without supervision. Every step below contains a revealed entry.
    indices = torch.nonzero(mask.any(1), as_tuple=True)[0]
    order_rng = torch.Generator(device=x.device).manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(len(indices), generator=order_rng, device=x.device)
        for rows in indices[order].split(bs):
            p = model(x[rows])
            observed = mask[rows]
            # Claude integration (red-team m1): per-target masked mean averaged over observed targets, as train_lf.py
            per_t = (p - y[rows]).square().masked_fill(~observed, 0.).sum(0) / observed.sum(0).clamp_min(1)
            loss = per_t[observed.any(0)].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite {decoder} loss at k={k}')
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    with torch.inference_mode():
        p = torch.cat([model(batch) for batch in q.split(bs)])
        p[:, ~mask.any(0)] = float('nan')
    return p


def knn(x, q, y, mask, k=50):
    """Exact cosine search in each marker's observed training bank."""
    x, q = F.normalize(x), F.normalize(q)
    p = x.new_full((len(q), y.shape[1]), float('nan'))
    for j in range(y.shape[1]):
        bank = x[mask[:, j]]
        if not len(bank):
            continue
        values = y[mask[:, j], j]
        # Bound the largest distance block to 64 MiB regardless of bank size.
        chunk = min(1024, max(1, 2**24 // len(bank)))
        for start in range(0, len(q), chunk):
            rows = slice(start, start + chunk)
            sim = q[rows] @ bank.T
            ix = sim.topk(min(k, len(bank)), sorted=False).indices
            p[rows, j] = values[ix].mean(1)
    return p


def check_split(z, targets):
    g, flags = z['G'], z['is_eval']
    if (g.ndim != 1 or not np.issubdtype(g.dtype, np.integer) or len(np.unique(g)) != len(g)
            or not len(g) or g.min() < 0 or g.max() >= len(targets['batch'])):
        raise ValueError('invalid or duplicate global row ids')
    if flags.shape != g.shape or not np.array_equal(np.unique(flags), [0, 1]):
        raise ValueError('need both train and eval rows indicated by binary is_eval')
    if not np.array_equal(z['names'].astype(str), targets['names']):
        raise ValueError('feature dump target names differ')
    source = ~flags.astype(bool)
    train_draw = np.unique(targets['batch'][g[source]])
    eval_draw = np.unique(targets['batch'][g[~source]])
    if (len(train_draw) != 1 or len(eval_draw) != 1
            or {train_draw.item(), eval_draw.item()} != {0, 1}):
        raise ValueError('source and eval rows must each belong to a single, different draw')
    return g, source, f'b{train_draw.item()}{eval_draw.item()}'


def dataset_name(file, default):
    if '=' in file:
        if default is not None:
            raise ValueError('choose either --feature-set or NAME=PATH')
        return file.split('=', 1)
    path = Path(file)
    stem = path.parent.name if path.stem == 'features' else path.stem
    # Actual n051 dumps live in <run>_b10/features.npz; explicit NAME= works for any layout.
    stem = re.sub(r'(?i)(?:^|[_-])(?:b10|b01|b1to0|b0to1)(?:$|(?=[_-]))', '', stem)
    return default or stem or 'features', file


def output_scores(p, g, t, label, decoder, k, direction):
    # Excluded targets can have no annotations in one draw.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return [dict(feature_set=label, decoder=decoder, k=str(k), direction=direction, **v)
                for v in target_scores(p, g, t['Y'], t['mask'], t['names'], range(len(t['names'])))]


def direction_scores(z, t, source, g, args):
    h = z['H']
    if h.ndim != 2 or len(h) != len(g) or h.shape[1] == 0:
        raise ValueError('H must be a nonempty row-aligned feature matrix')
    def tensor(a):
        return torch.as_tensor(a, device=args.device)
    raw = tensor(h[source]).float()
    raw_eval = tensor(h[~source]).float()
    del h
    if not (torch.isfinite(raw).all() and torch.isfinite(raw_eval).all()):
        raise ValueError('nonfinite features')
    tr = g[source]
    mask = tensor(t['mask'][tr])
    y = tensor(t['Y'][tr]).float().masked_fill(~mask, 0.)
    if not torch.isfinite(y).all() or not mask.any():
        raise ValueError('invalid observed source labels')

    mu, sd = scaler(raw)
    x, queries = (raw - mu) / sd, (raw_eval - mu) / sd
    del raw_eval
    fit, val = inner_split(t['slide'][tr], t['fov'][tr], args.seed)
    yield 'ridge', '-', ridge(raw, x, queries, y, mask, tensor(fit), tensor(val))
    del raw
    for method, k in [('reduced_rank', k) for k in RANKS] + [('mlp', '-')]:
        yield method, k, train_network(x, queries, y, mask, method, k, seed=args.seed,
                                       epochs=args.epochs, bs=args.bs, lr=args.lr)
    yield 'knn', 50, knn(x, queries, y, mask)


def assert_reference(actual, key):
    if not np.isfinite(actual) or abs(actual - RECORDED[key]) > 1e-4:
        raise AssertionError(f'reference {key} = {actual:.8f}, expected {RECORDED[key]:.4f} +/- 1e-4')


def run(args):
    with np.load(args.targets) as data:
        t = {k: data[k] for k in ('names', 'Y', 'mask', 'slide', 'fov', 'batch')}
    t['names'] = t['names'].astype(str)
    if (t['mask'].dtype != np.bool_ or t['Y'].shape != t['mask'].shape
            or t['Y'].shape != (len(t['batch']), 13) or len(t['names']) != 13):
        raise ValueError('expected 13 marker targets and aligned boolean mask')
    c11 = t['names'][common11(t['names'])]
    if len(c11) != 11 or len(set(t['names'])) != 13:
        raise ValueError('expected 11 common targets and unique names')
    rows, pairs = [], {}
    # Read only small metadata and P on the first pass. Fail before expensive fits.
    for spec in args.feats:
        label, path = dataset_name(spec, args.feature_set)
        with np.load(path) as z:
            g, source, d = check_split(z, t)
            if d in pairs.setdefault(label, {}):
                raise ValueError(f'duplicate {label} {d}')
            has_p = 'P' in z.files
            pairs[label][d] = (path, has_p)
            if has_p:
                p = z['P']
                if p.shape != (len(g), 13) or not np.isfinite(p[~source]).all():
                    raise ValueError('invalid reference predictions')
                result = output_scores(p[~source], g[~source], t, label, 'reference', '-', d)
                reference = pd.DataFrame(result).query('target in @c11')
                if not np.isfinite(reference.r).all():
                    raise AssertionError(f'missing common-11 reference score for {d}')
                if 'n051_v10_base' in str(path): assert_reference(aggregate_goal(reference, ['feature_set'])[d].item(), d)   # red-team m3
                rows.extend(result)
    for label, files in pairs.items():
        if set(files) != {'b10', 'b01'} or files['b10'][1] != files['b01'][1]:
            raise ValueError(f'{label}: need both directions with matching presence of reference P')
        if files['b10'][1] and 'n051_v10_base' in str(files['b10'][0]):
            refs = pd.DataFrame(rows).query('feature_set == @label and target in @c11')
            assert_reference(aggregate_goal(refs, ['feature_set']).GOAL.item(), 'GOAL')
        elif args.reference_only:
            raise ValueError(f'{label}: --reference-only requires P')

    if not args.reference_only:
        for label, files in pairs.items():
            for d, (path, _) in files.items():
                with np.load(path) as z:
                    g, source, _ = check_split(z, t)
                    for decoder, k, p in direction_scores(z, t, source, g, args):
                        rows.extend(output_scores(p.cpu().numpy(), g[~source], t, label, decoder, k, d))
                        print(label, d, decoder, k, flush=True)
    df = pd.DataFrame(rows)
    keys = ['feature_set', 'decoder', 'k']
    common = df[df.target.isin(c11)]
    goal = aggregate_goal(common, keys).set_index(keys)
    # For non-reference decoders a missing/constant prediction is observable in decoders.csv.
    # Do not promote the mean over 10 (or fewer) targets as a valid common-11 GOAL.
    counts = common.groupby(keys + ['direction']).r.count().unstack('direction')
    for d in ('b10', 'b01'):
        goal.loc[counts[d] != 11, [d, 'GOAL']] = np.nan
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / 'decoders.csv', index=False)
    goal.reset_index()[keys + ['b10', 'b01', 'GOAL']].to_csv(out / 'goal.csv', index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('feats', nargs='+', help='paths for a paired set or repeated NAME=PATH')
    ap.add_argument('--feature-set', help='label for one paired set')
    ap.add_argument('--targets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--bs', type=int, default=512)
    ap.add_argument('--lr', type=float, default=.001)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--reference-only', action='store_true')
    args = ap.parse_args()
    if args.epochs < 1 or args.bs < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        ap.error('epochs, bs and lr must be positive')
    run(args)


if __name__ == '__main__':
    main()
