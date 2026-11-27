"""Two optimizer steps per arm/direction plus controls for masks, scoring, and row identity."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
import n087_decoders as dec


def fixture(tmp):
    rng = np.random.default_rng(18)
    n = 80
    batch = np.repeat([0, 1], n)
    x = rng.normal(size=(2*n, 7)).astype(np.float32)
    y = x @ rng.normal(size=(7, 13)).astype(np.float32)
    mask = rng.random(y.shape) > .15
    names = np.array([f'm{i}' for i in range(8)] + ['grp:CD123/HLA'] + [f'm{i}' for i in range(9, 13)])
    # No source observation for a non-common target, and poison unobserved entries.
    mask[batch == 1, 8] = False
    y[~mask] = np.nan
    t = dict(Y=y, mask=mask, names=names, batch=batch, slide=np.tile([0, 1], n),
             fov=np.tile(np.arange(n)//4, 2))
    np.savez(tmp / 'targets.npz', **t)
    paths = []
    for src, direction in [(1, 'b10'), (0, 'b01')]:
        g = rng.permutation(len(batch))
        ev = batch[g] != src
        p = np.zeros(y.shape)
        for j in range(13):
            observed = ev & mask[g, j]
            if not observed.any():
                continue
            truth = y[g[observed], j].astype(float)
            truth -= truth.mean()
            truth /= np.linalg.norm(truth)
            noise = rng.normal(size=len(truth))
            noise -= noise.mean()
            noise -= truth.dot(noise) * truth
            noise /= np.linalg.norm(noise)
            r = dec.RECORDED[direction]
            p[observed, j] = r*truth + np.sqrt(1-r*r)*noise
        path = tmp / f'base_{direction}.npz'
        np.savez(path, H=x[g], P=p, G=g, is_eval=ev, names=names)
        paths.append(path)
    return paths, t


def test_cli_two_steps_per_arm(tmp_path):
    files, _ = fixture(tmp_path)
    subprocess.run([sys.executable, '-IB', str(SCRIPTS / 'n087_decoders.py'), *map(str, files),
                    '--targets', str(tmp_path / 'targets.npz'), '--out', str(tmp_path),
                    '--epochs', '1', '--bs', '40', '--device', 'cpu'], check=True)
    long, goal = (pd.read_csv(tmp_path / f) for f in ('decoders.csv', 'goal.csv'))
    assert long.shape == (13*13*2, 6)
    assert list(goal.columns) == ['feature_set', 'decoder', 'k', 'b10', 'b01', 'GOAL']
    assert goal.feature_set.unique().tolist() == ['base']
    ref = goal.set_index('decoder').loc['reference']
    assert ref.GOAL == pytest.approx((.6867 + .6592)/2)
    assert goal.shape == (13, 6) and np.isfinite(goal.GOAL).all()
    excluded = long[(long.direction == 'b10') & (long.target == 'grp:CD123/HLA') & (long.decoder != 'reference')]
    assert excluded.r.isna().all()


def test_fail_closed(tmp_path):
    files, t = fixture(tmp_path)
    args = SimpleNamespace(targets=tmp_path / 'targets.npz', reference_only=True, feature_set=None,
                           feats=files, out=tmp_path)
    args.feats = [str(files[0])]
    with pytest.raises(ValueError, match='both directions'):
        dec.run(args)
    args.feats = list(map(str, files))
    with np.load(files[0]) as z:
        d = dict(z)
    d['P'] *= -1
    np.savez(files[0], **d)
    with pytest.raises(AssertionError, match='reference b10'):
        dec.run(args)
    d['is_eval'][0] = ~d['is_eval'][0]
    with pytest.raises(ValueError, match='different draw'):
        dec.check_split(d, t)
    d['G'][0] = d['G'][1]
    with pytest.raises(ValueError, match='global row ids'):
        dec.check_split(d, t)


def test_fov_groups_and_inductive_predictions(tmp_path):
    files, t = fixture(tmp_path)
    with np.load(files[0]) as f:
        z = dict(f)
    g, source, _ = dec.check_split(z, t)
    tr = g[source]
    ids = np.column_stack((t['slide'][tr], t['fov'][tr]))
    fit, val = dec.inner_split(*ids.T, 42)
    assert len(set(map(tuple, ids[val]))) == 4
    assert set(map(tuple, ids[fit])).isdisjoint(set(map(tuple, ids[val])))
    args = SimpleNamespace(device='cpu', seed=0, bs=40, epochs=1, lr=.001)
    def predictions():
        return {(name,str(k)): p.numpy() for name,k,p in dec.direction_scores(z, t, source, g, args)}
    result = predictions()
    z['H'][np.flatnonzero(~source)[0]] = 1000
    t['Y'][g[~source]] = np.nan
    t['Y'][~t['mask']] = 928
    changed = predictions()
    for key in result:
        np.testing.assert_allclose(changed[key][1:], result[key][1:], atol=0, rtol=0, equal_nan=True)


def test_closed_form_and_exact_neighbors():
    x = torch.tensor([[-1., 0.], [1., 3.], [1., 2.], [2., 0.], [1., -3.]])
    y = torch.tensor([1., 2., 3., 4., 5.])
    q = torch.tensor([[1., 0.], [0., 1.]])
    a = torch.tensor([.1, 1., 3.])
    centered = x - x.mean(0)
    expected = torch.stack([(q-x.mean(0)) @ torch.linalg.solve(
        centered.T @ centered + len(x)*c*torch.eye(2), centered.T@(y-y.mean())) + y.mean() for c in a], 1)
    torch.testing.assert_close(dec.ridge_path(x, y, q, a), expected)
    y = torch.stack([y, y], dim=1)
    mask = torch.ones_like(y).bool()
    mask[3, 1] = False
    p = dec.knn(x, q, y, mask, k=1)
    torch.testing.assert_close(p, torch.tensor([[4., 3.], [2., 2.]]))


def test_explicit_pair_without_p(tmp_path):
    files, _ = fixture(tmp_path)
    for path in files:
        with np.load(path) as data:
            arrays = {key: data[key] for key in data.files if key != 'P'}
        np.savez(path, **arrays)
    args = SimpleNamespace(targets=tmp_path / 'targets.npz', reference_only=False, feature_set=None,
                           feats=[f'AE64={p}' for p in files], out=tmp_path, device='cpu',
                           bs=40, epochs=1, seed=0, lr=.001)
    dec.run(args)
    scores = pd.read_csv(tmp_path / 'goal.csv')
    assert scores.feature_set.unique().tolist() == ['AE64']
    assert len(scores) == 12 and 'reference' not in set(scores.decoder)
