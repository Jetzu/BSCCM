"""CPU comparisons with the archived N080 trainer and analysis."""
import importlib.util
from itertools import combinations
import json
import os
from pathlib import Path
import runpy
import sys

import numpy as np
import pandas as pd
import pytest
import torch

HERE = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('BSCCM_REPO', HERE.parents[3]))
N080 = REPO / '.agent-research/results/N080-joint-decoding-sandbox-on-s-biad452-all'
sys.path.insert(0, str(HERE))
import n087_sandbox_eval as evaluate
import train_ifc_sandbox as trainer


def source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def pack(tmp_path_factory):
    fixture = source_module('sandbox_fixtures', N080 / 'codex/tests/make_fixture.py')
    return fixture.make_fixture(tmp_path_factory.mktemp('marker-pack'))


def train(module, pack, out, kind=None, extra=()):
    args = ['--regime', 'single', '--test-donor', '0', '--pack', str(pack), '--out', str(out),
            '--device', 'cpu', '--threads', '1', '--workers', '0', '--epochs', '1', '--bs', '50',
            '--knn-impute', '0', '--smoke-tiny-model']
    if kind:
        args += ['--marker-prior', kind, '--prior-dim', '2']
    module.main([*args, *extra])
    return torch.load(out / 'ckpt.pt', map_location='cpu', weights_only=False)


def test_default_control_is_bit_identical(pack, tmp_path):
    baseline = source_module('original_trainer', REPO / 'scripts/train_ifc_sandbox.py')
    a = train(baseline, pack, tmp_path / 'original')
    b = train(trainer, pack, tmp_path / 'copy', 'none')
    assert a['cfg'] == b['cfg'] and a['history'] == b['history']
    for k in a['model']:
        assert torch.equal(a['model'][k], b['model'][k])
    for name in ['setup.json', 'metrics.json', 'eval_preds.npz']:
        assert (tmp_path / 'original' / name).read_bytes() == (tmp_path / 'copy' / name).read_bytes()


@pytest.mark.parametrize('kind', ['linear', 'ae'])
def test_only_validation_profiles_enter_frozen_prior(kind, tmp_path, pack, monkeypatch):
    fit = trainer.marker_prior
    profiles = []
    fitted = []
    def capture(kind, dim, val_y, seed, device):
        profiles.append(val_y.copy())
        decoder = fit(kind, dim, val_y, seed, device)
        assert all(not p.requires_grad for p in decoder.parameters())
        fitted.append({k: v.clone() for k, v in decoder.state_dict().items()})
        return decoder
    monkeypatch.setattr(trainer, 'marker_prior', capture)
    a = train(trainer, pack, tmp_path / 'a', kind)
    assert a['step'] == 2
    for k, v in fitted[0].items():
        assert torch.equal(v, a['model']['prior.' + k])
    with np.load(pack / 'labels.npz') as f:
        data = dict(f)
    tr, va, te = trainer.donor_split(data['sample'], 0)
    mask, _ = trainer.reveal_masks(np.arange(len(data['sample'])), 'single', 42)
    src = data['I_raw']
    cofactor = np.array([trainer.robust_scale(src[tr, j][mask[tr, j]]) for j in range(9)], np.float32)
    expected = np.arcsinh(src[va] / cofactor)
    for name in np.unique(data['sample'][va]):
        ids = np.flatnonzero(data['sample'][va] == name)
        for j in range(9):
            ref = expected[ids[mask[va[ids], j]], j]
            if len(ref) >= 2:
                expected[ids, j] = (expected[ids, j] - float(ref.mean())) / (float(ref.std()) + 1e-6)
    np.testing.assert_allclose(profiles[0], expected, atol=1e-5)
    src[te] -= 5e4
    src[tr] = np.where(mask[tr], src[tr], src[tr] + 2e4)
    altered = tmp_path / 'altered_pack'
    altered.mkdir()
    (altered / 'lf.npy').symlink_to(pack / 'lf.npy')
    np.savez(altered / 'labels.npz', **data)
    b = train(trainer, altered, tmp_path / 'b', kind)
    assert np.array_equal(profiles[0], profiles[1])
    assert a['history'] == b['history']
    assert a['model']['mean.weight'].shape[0] == 2
    for k, v in a['model'].items():
        assert torch.equal(v, b['model'][k])


def test_ae_resume_restores_prior_without_refitting(pack, tmp_path, monkeypatch):
    whole = train(trainer, pack, tmp_path / 'whole', 'ae')
    partial = train(trainer, pack, tmp_path / 'resume', 'ae', ['--stop-after-steps', '1'])
    assert partial['step'] == 1
    original = trainer.marker_prior
    def resume_only(kind, dim, val_y, seed, device):
        assert val_y is None
        prior = original(kind, dim, val_y, seed, device)
        assert not torch.equal(partial['model']['prior.decoder.0.weight'], prior.decoder[0].weight)
        return prior
    monkeypatch.setattr(trainer, 'marker_prior', resume_only)
    final = train(trainer, pack, tmp_path / 'resume', 'ae')
    assert whole['history'] == final['history']
    for k in whole['model']:
        assert torch.equal(whole['model'][k], final['model'][k])


def test_pca_projects_onto_leading_standardized_subspace():
    val = np.random.default_rng(20).normal(size=(30, 9)).astype(np.float32)
    val[:, 1] += 0.3 * val[:, 2]
    d = trainer.marker_prior('linear', 2, val, 3, 'cpu')
    z = (val - val.mean(0)) / val.std(0)
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    u = d.decoder.weight.detach().numpy()
    np.testing.assert_allclose(u @ u.T, vt[:2].T @ vt[:2], atol=3e-6)


def test_arbitrary_run_matches_original_posthoc(tmp_path, monkeypatch):
    # Large enough to exercise the historical >=100-cell threshold using labels alone.
    rng = np.random.default_rng(42)
    n = 1500
    M = trainer.MARKERS
    pack = tmp_path / 'pack'
    runs = tmp_path / 'runs'
    refout = tmp_path / 'ref'
    out = tmp_path / 'out'
    pack.mkdir()
    runs.mkdir()
    raw = rng.normal(3000, 700, (3*n, 9))
    for donor in range(3):
        idx = np.arange(n) + donor*n
        for marker, sl in [('CD15', slice(0, 400)), ('CD14', slice(400, 600)),
                           ('CD3', slice(600, 1200)), ('CD4', slice(600, 900)),
                           ('CD8', slice(900, 1200)), ('CD19', slice(1200, 1350)),
                           ('CD56', slice(1350, None))]:
            raw[idx[sl], M.index(marker)] += 40000
    samples = np.repeat(['sample0', 'sample1', 'sample2'], n)
    np.savez(pack / 'labels.npz', I_raw=raw, sample=samples, names=M)
    other = tmp_path / 'folder with spaces'
    other.mkdir()
    (runs / 'n080_full_d0').symlink_to(other)
    test = np.arange(n)
    ya = np.arcsinh(raw[test] / 1000)
    p = (ya + rng.normal(0, .1, ya.shape)).astype(np.float32)
    np.savez(other / 'eval_preds.npz', rows=test, Y=ya, P=p, P_knn=p + 1,
             sample=samples[test], names=M)
    biases = {f'{a}--{b}': {'predicted_minus_true': (-1)**i * (i + 1) / 40}
              for i, (a, b) in enumerate(combinations(M, 2))}
    met = dict(names=M, test_donor=0, n_test=n, model=dict(pairwise_correlation_bias=biases))
    (other / 'metrics.json').write_text(json.dumps(met))
    src = N080 / 'n080_eval.py'
    monkeypatch.setattr(sys, 'argv', [str(src), str(runs), str(pack), str(refout)])
    runpy.run_path(str(src))
    evaluate.main([f'full={other}', '--out', str(out), '--pack', str(pack)])
    for name in ['marker', 'withintype', 'pairs', 'populations', 'summary']:
        old = pd.read_csv(refout / f'n080_eval_{name}.csv')
        new = pd.read_csv(out / f'n087_eval_{name}.csv')[old.columns]
        pd.testing.assert_frame_equal(old, new, atol=5.1e-5 if name == 'summary' else 1e-12,
                                      check_exact=False, rtol=0)
    registered = pd.read_csv(out / 'n087_eval_36pairs.csv')
    assert registered['n_defined'].tolist() == [36]
    assert registered['mean_abs_bias'].iloc[0] == pytest.approx(18.5 / 40)
    biases['CD15--Siglec8']['predicted_minus_true'] = None
    assert evaluate.registered_metric(met)['n_defined'] == 35
    assert np.isnan(evaluate.registered_metric(met)['mean_abs_bias'])
