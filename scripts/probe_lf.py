"""Frozen LF encoder probe; identical split, sampling and heads for SSL/supervised."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import csv
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import timm
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from bsccm.data import W
from bsccm.gpu_norm import RawSet, GPUNorm


@torch.no_grad()
def extract(model, rows, kind, bs, workers):
    kw = dict(num_workers=workers, pin_memory=True)
    if workers:
        kw.update(persistent_workers=False, prefetch_factor=6)
    loader = torch.utils.data.DataLoader(RawSet(rows), batch_size=bs, **kw); gnorm = GPUNorm('cuda')
    features = np.empty((len(rows), model.num_features), dtype=np.float32)
    offset = 0
    for raw, ph, g in loader:
        x = gnorm(raw.cuda(non_blocking=True), ph.cuda(non_blocking=True), g.cuda(non_blocking=True), 'all', True)
        if kind == 'ssl':
            x = torch.cat((x, torch.ones_like(x)), dim=1)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model(x.contiguous(memory_format=torch.channels_last))
        h = h.float().cpu().numpy()
        if h.shape != (len(x), model.num_features) or not np.isfinite(h).all():
            raise ValueError('encoder must return finite pooled (batch,num_features) features')
        features[offset:offset + len(x)] = h
        offset += len(x)
    if offset != len(rows):
        raise RuntimeError('feature extraction row count mismatch')
    return features


def pearson(p, y):
    """Same centred Pearson and epsilon convention as train_lf.evaluate."""
    p, y = np.asarray(p, dtype=np.float64), np.asarray(y, dtype=np.float64)
    p, y = p - p.mean(), y - y.mean()
    return float((p * y).sum() / (np.sqrt((p * p).sum() * (y * y).sum()) + 1e-9))


def ridge_path_predict(x, y, ev, alphas):
    """Closed-form ridge path with intercept, one eigensolve for all alphas.

    Fit scaling here, within EACH CV fold. Dual solve if n<d avoids a large
    singular primal system on sparsely labelled targets. Ridge objective is
    sum squared residuals + alpha * squared coefficient norm (sklearn units).
    """
    scaler = StandardScaler()
    x = scaler.fit_transform(np.asarray(x, dtype=np.float64))
    ev = scaler.transform(np.asarray(ev, dtype=np.float64))
    y = np.asarray(y, dtype=np.float64)
    mean = y.mean()
    y = y - mean
    if x.shape[0] >= x.shape[1]:
        eig, q = np.linalg.eigh(x.T @ x)
        rhs = q.T @ (x.T @ y)
        coef = q @ (rhs[:, None] / (eig.clip(0)[:, None] + alphas))
    else:
        eig, q = np.linalg.eigh(x @ x.T)
        rhs = q.T @ y
        coef = (x.T @ q) @ (rhs[:, None] / (eig.clip(0)[:, None] + alphas))
    return ev @ coef + mean


def ridge_cv(x, y, ev, seed=0):
    if len(y) < 5:
        raise ValueError('each marker needs at least five labelled training rows for 5-fold CV')
    alphas = np.logspace(-2, 4, 7)
    mse = []
    for tr, va in KFold(5, shuffle=True, random_state=seed).split(x):
        pred = ridge_path_predict(x[tr], y[tr], x[va], alphas)
        mse.append(((pred - y[va, None]) ** 2).mean(0))
    mse = np.mean(mse, axis=0)
    j = int(np.argmin(mse))
    pred = ridge_path_predict(x, y, ev, alphas[j:j + 1])[:, 0]
    if not np.isfinite(pred).all():
        raise FloatingPointError('nonfinite ridge predictions')
    return pred, float(alphas[j]), mse.tolist()


def evaluate_groups(pred, y, mask, cls3, names):
    """Baseline's test-only class oracle and gated/within-class thresholds.

    The oracle deliberately uses TEST truth class means; it is descriptive, not
    a fitted predictor and never feeds feature extraction, CV or a probe head.
    """
    metrics = {}
    groups = [j for j, n in enumerate(names) if n.startswith('grp:')]
    if len(groups) != 4:
        raise ValueError('expected exactly four grp: targets')
    for j in groups:
        v = (cls3 >= 0) & mask[:, j]
        metrics[f'n_gated/{names[j]}'] = int(v.sum())
        for key in ('oracle_classmean_r', 'r_gated', 'r_within_lym', 'r_within_gran', 'r_within_mono'):
            metrics[f'{key}/{names[j]}'] = None
        if v.sum() < 50:
            continue
        means = {k: y[v & (cls3 == k), j].mean() for k in np.unique(cls3[v])}
        oracle = np.array([means[k] for k in cls3[v]])
        metrics[f'oracle_classmean_r/{names[j]}'] = pearson(oracle, y[v, j])
        metrics[f'r_gated/{names[j]}'] = pearson(pred[v, j], y[v, j])
        for k, kn in enumerate(['lym', 'gran', 'mono']):
            vk = v & (cls3 == k)
            metrics[f'n_within_{kn}/{names[j]}'] = int(vk.sum())
            if vk.sum() > 30:
                metrics[f'r_within_{kn}/{names[j]}'] = pearson(pred[vk, j], y[vk, j])
    return metrics


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--kind', choices=['ssl', 'sup'], required=True)
    p.add_argument('--arch', default='resnet50')
    p.add_argument('--out', required=True)
    p.add_argument('--n-train', type=int, default=60000)
    p.add_argument('--n-test', type=int, default=20000)
    p.add_argument('--workers', type=int, default=15)
    p.add_argument('--bs', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--all-feats', action='store_true', help='also export features of ALL cells (lf_feats_all.npy, f16) for stage 2')
    p.add_argument('--cpus', type=int, default=48, help='BLAS threads for each closed-form solve / logistic fit')
    a = p.parse_args()
    if min(a.n_train, a.n_test, a.bs, a.cpus) < 1 or a.workers < 0 or a.seed < 0:
        p.error('invalid row count, batch size, workers, seed or CPU count')
    return a


def main():
    a = parse_args()
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('feature extraction requires a CUDA device with bf16 support')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(W + '/targets_v3.npz') as t:
        y, mask, cls3 = t['Y'], t['mask'], t['cls3']
        split, hist = t['split'], t['hist_row']
        names = [str(n) for n in t['names']]
    if len(names) != 13 or names.count('autofl') != 1:
        raise ValueError('expected 13 targets including exactly one autofl')
    has = mask.any(1) | (cls3 >= 0)
    rng = np.random.default_rng(a.seed)
    tr = rng.permutation(np.flatnonzero((split == 0) & has))[:a.n_train]
    te = rng.permutation(np.flatnonzero((split == 2) & has))[:a.n_test]
    if len(tr) != a.n_train or len(te) != a.n_test:
        raise ValueError('requested more eligible rows than available; choose explicit smaller counts')
    hr = np.flatnonzero(hist >= 0)  # global-index order, saved indices define alignment
    if not len(hr):
        raise ValueError('no Giemsa-paired cells found')
    all_rows = np.union1d(np.union1d(tr, te), hr)
    model = timm.create_model(a.arch, pretrained=False, in_chans=48 if a.kind == 'ssl' else 24,
                              num_classes=0)
    s = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    if a.kind == 'ssl':
        state = s  # encoder_final.pt is exactly an encoder state_dict
    else:
        # train_lf.py writes a ModuleDict state_dict under 'm' (enc.*, reg.*, c3.*, c10.*).
        state = {k[len('enc.'):]: v for k, v in s['m'].items() if k.startswith('enc.')}
        if not state:
            raise ValueError('supervised checkpoint has no m/enc.* encoder weights')
    model.load_state_dict(state, strict=True)
    del s, state
    model.requires_grad_(False)
    model = model.cuda().to(memory_format=torch.channels_last).eval()
    features = extract(model, all_rows, a.kind, a.bs, a.workers)
    if a.all_feats:   # stage-2 input: frozen features for every cell, row = global index
        np.save(out / 'lf_feats_all.npy', extract(model, np.arange(len(split)), a.kind, a.bs, a.workers).astype(np.float16), allow_pickle=False)
    del model
    torch.cuda.empty_cache()
    xtr, xte = features[np.searchsorted(all_rows, tr)], features[np.searchsorted(all_rows, te)]
    np.save(out / 'lf_feats_hist.npy', features[np.searchsorted(all_rows, hr)], allow_pickle=False)
    np.save(out / 'lf_feats_hist_indices.npy', hr, allow_pickle=False)
    np.save(out / 'lf_feats_hist_hist_row.npy', hist[hr], allow_pickle=False)
    np.savez(out / 'probe_rows.npz', train=tr, test=te)
    del features
    predictions = np.empty((len(te), len(names)), dtype=np.float64)
    metrics, cv = {}, {}
    # Every dense solve uses the requested CPU fraction; folds run sequentially
    # to avoid nested BLAS oversubscription and multiplying peak RAM by five.
    with threadpool_limits(limits=a.cpus), warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for j, name in enumerate(names):
            train_mask = mask[tr, j]
            test_mask = mask[te, j]
            if not np.isfinite(y[tr[train_mask], j]).all() or not np.isfinite(y[te[test_mask], j]).all():
                raise ValueError(f'nonfinite observed target: {name}')
            pred, alpha, mse = ridge_cv(xtr[train_mask], y[tr[train_mask], j], xte, a.seed)
            predictions[:, j] = pred
            metrics[f'n_train/{name}'] = int(train_mask.sum())
            metrics[f'n/{name}'] = int(test_mask.sum())
            metrics[f'r/{name}'] = pearson(pred[test_mask], y[te[test_mask], j]) if test_mask.sum() > 20 else None
            metrics[f'alpha/{name}'] = alpha
            cv[name] = {'alphas': np.logspace(-2, 4, 7).tolist(), 'mse': mse}
        marker_rs = [metrics[f'r/{name}'] for name in names if name != 'autofl']
        if any(r is None for r in marker_rs):
            raise ValueError('cannot report a mean over 12 markers: a test marker has <=20 observations')
        metrics['r_mean_markers'] = float(np.mean(marker_rs))
        labtr, labte = cls3[tr] >= 0, cls3[te] >= 0
        classes, counts = np.unique(cls3[tr[labtr]], return_counts=True)
        if not np.array_equal(classes, np.arange(3)) or (counts < 5).any():
            raise ValueError('need >=5 training rows in each cls3 class for stratified 5-fold CV')
        if not labte.any():
            raise ValueError('no cls3-labelled test rows')
        logistic = make_pipeline(StandardScaler(), LogisticRegression(solver='lbfgs', max_iter=2000))
        search = GridSearchCV(logistic, {'logisticregression__C': np.logspace(-3, 3, 7)},
                              cv=StratifiedKFold(5, shuffle=True, random_state=a.seed),
                              scoring='balanced_accuracy', n_jobs=1, error_score='raise')
        search.fit(xtr[labtr], cls3[tr[labtr]])
        cpred = search.predict(xte[labte])
        metrics['cls3_acc'] = float(accuracy_score(cls3[te[labte]], cpred))
        metrics['cls3_balanced_acc'] = float(balanced_accuracy_score(cls3[te[labte]], cpred))
        metrics['cls3_C'] = float(search.best_params_['logisticregression__C'])
        metrics['cls3_n_train'] = int(labtr.sum())
        metrics['cls3_n_test'] = int(labte.sum())
        cv['cls3'] = {'C': search.cv_results_['param_logisticregression__C'].tolist(),
                      'mean_balanced_acc': search.cv_results_['mean_test_score'].tolist()}
    metrics.update(evaluate_groups(predictions, y[te], mask[te], cls3[te], names))
    result = dict(args=vars(a), metrics=metrics, cv=cv, names=names,
                  n_train=len(tr), n_test=len(te), n_hist=len(hr),
                  notes=['SSL rows=all is transductive to test FOV images, but no labels enter SSL.',
                         'Oracle class means use test truth, matching train_lf.evaluate; descriptive ceiling only.',
                         'Baseline thresholds: r needs >20 rows, gated >=50, within-class >30; null means insufficient rows.',
                         'Ridge CV selects mean fold MSE; logistic CV selects balanced accuracy; scaling is fold-local.'])
    with (out / 'probe.json').open('w') as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (out / 'probe.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['metric', 'value'])
        writer.writerows(metrics.items())
    print(json.dumps(metrics, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
