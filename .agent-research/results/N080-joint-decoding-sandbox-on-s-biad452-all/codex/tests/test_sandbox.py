import json
import math
import os
import signal

import numpy as np
import pytest
import torch

import train_ifc_sandbox as s
from tests.make_fixture import make_fixture


@pytest.fixture(scope='session')
def pack(tmp_path_factory):
    return make_fixture(tmp_path_factory.mktemp('pack'))


@pytest.mark.parametrize('regime', ['full', 'single', 'single_pooled'])
def test_reveal_masks(regime):
    rows = np.arange(10000)
    mask, pool = s.reveal_masks(rows, regime, 42)
    assert np.array_equal(mask[77:121], s.reveal_masks(rows[77:121], regime, 42)[0])
    assert np.array_equal(pool[77:121], s.reveal_masks(rows[77:121], regime, 42)[1])
    if regime == 'full':
        assert mask.all() and not pool.any()
    else:
        pooled = pool.any(1)
        assert (mask[~pooled].sum(1) == 1).all()
        assert not mask[pooled].any()
        assert (pool[:, 0] == pool[:, 1]).all()
        assert .085 < pooled.mean() < .115 if regime == 'single_pooled' else not pooled.any()
        counts = mask.sum(0)
        assert counts.min() > .8 * counts.mean()


@pytest.mark.parametrize('donor', [0, 1, 2])
def test_donor_split(pack, donor):
    samples = np.load(pack / 'labels.npz')['sample']
    tr, va, te = s.donor_split(samples, donor)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    assert len(set(tr) | set(va) | set(te)) == len(samples)
    d = s.donor_ids(samples)
    assert set(d[tr]) == {(donor + 2) % 3}
    assert set(d[va]) == {(donor + 1) % 3}
    assert set(d[te]) == {donor}
    with pytest.raises(ValueError):
        s.donor_ids(['sample3'])


def test_hidden_labels_have_zero_loss_and_gradient():
    pred = torch.randn(5, 9, requires_grad=True)
    target = torch.randn(5, 9)
    mask = torch.zeros(5, 9, dtype=torch.bool)
    mask[range(5), range(5)] = True
    loss = s.masked_mse(pred, target, mask)
    changed = target.clone()
    changed[~mask] = float('nan')
    assert torch.equal(loss, s.masked_mse(pred, changed, mask))
    loss.backward()
    assert (pred.grad[~mask] == 0).all()
    assert torch.equal(s.masked_mse(pred, changed, torch.zeros_like(mask)), torch.tensor(0.))


def test_pooled_only_loss_cannot_see_single_targets():
    model = s.SandboxModel(pooled_mode='head', tiny=True)
    output = model(torch.randn(4, 3, 128, 128))
    ids = torch.arange(4)
    y = torch.full((4, 9), float('nan'))
    mask = torch.zeros(4, 9, dtype=torch.bool)
    pools, pmask = torch.randn(4, 2), torch.ones(4, 2, dtype=torch.bool)
    args = s.parse_args(['--out', '/tmp/n080_codex/unused', '--test-donor', '0', '--device', 'cpu'])
    loss = s.batch_loss(model, output, ids, y, mask, pools, pmask, y, None, args)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.count_nonzero(model.mean.weight.grad) == 0
    assert torch.count_nonzero(model.pool_head.weight.grad) > 0


def test_pool_formula_calibration_and_no_hidden_map_labels(pack):
    data = np.load(pack / 'labels.npz')
    y, raw = data['Y'], data['I_raw']
    mask, pools = s.reveal_masks(np.arange(len(y)), 'single_pooled', 42)
    cal = s.fit_pool_calibration(raw, y, mask, pools)
    changed_y = y.copy()
    changed_y[~mask] = 99999
    changed_cal = s.fit_pool_calibration(raw, changed_y, mask, pools)
    assert cal == changed_cal
    # Alter unobserved RAW components only on single-stain rows.
    changed_raw = raw.copy()
    changed_raw[(~mask) & (~pools[:, :1])] = -1e6
    assert cal == s.fit_pool_calibration(changed_raw, y, mask, pools)
    inverse = s.MonotoneInverse(cal)
    output = inverse(torch.from_numpy(y)).detach().numpy()
    reconstructed_raw = cal['c'] * np.sinh(inverse.transformed(torch.from_numpy(y)).detach().numpy())
    assert np.allclose(output, s.pooled_targets(reconstructed_raw, cal), atol=2e-5)
    expected = s.pooled_targets(raw, cal)
    assert np.allclose(expected[pools[:, 0]].mean(0), 0, atol=1e-6)
    assert np.allclose(expected[pools[:, 0]].std(0), 1, atol=1e-6)
    low, high = torch.zeros(1, 9), torch.ones(1, 9)
    assert torch.all(inverse.transformed(high) > inverse.transformed(low))


def test_metrics_identity_and_known_pair_bias():
    rng = np.random.default_rng(54)
    y = rng.normal(size=(3000, 9))
    gates = s.fit_gates(y)
    metrics = s.compute_metrics(y, y.copy(), gates)
    assert metrics['macro_r'] == pytest.approx(1)
    assert metrics['vector_rmse_mean'] == 0
    assert metrics['vector_cosine_mean'] == pytest.approx(1)
    assert metrics['coexpression']['overall']['frobenius'] == 0
    assert len(metrics['pairwise_correlation_bias']) == 36
    for name, result in metrics['gating'].items():
        assert result['f1'] in (1, None), name
    for pop in metrics['within_population'].values():
        if pop['n'] >= 3:
            assert pop['macro_r'] == pytest.approx(1)
    assert metrics['population_means']['macro_r'] == pytest.approx(1)
    p = y.copy()
    p[:, 1] = p[:, 0]
    delta = s.compute_metrics(y, p, gates)['pairwise_correlation_bias']['CD15--Siglec8']
    assert delta['predicted_minus_true'] == pytest.approx(1 - np.corrcoef(y[:, 0], y[:, 1])[0, 1])
    zero = s.compute_metrics(y, np.zeros_like(y), gates)
    assert zero['macro_r'] is None and zero['vector_cosine_mean'] is None
    assert zero['coexpression']['overall']['frobenius'] is None
    json.dumps(zero, allow_nan=False)


def test_cluster_correct_does_not_mean_cell_correct():
    rng = np.random.default_rng(11)
    y = rng.normal(size=(2000, 9))
    gates = s.fit_gates(y)
    p = np.zeros_like(y)
    for mask in s.coarse_populations(y, gates).values():
        p[mask] = y[mask].mean(0)
    result = s.compute_metrics(y, p, gates)
    assert result['population_means']['macro_r'] == pytest.approx(1)
    for pop in result['within_population'].values():
        assert pop['macro_r'] is None


def test_fscore_and_double_positive():
    truth = np.array([1, 1, 0, 0], bool)
    predicted = np.array([1, 0, 1, 0], bool)
    assert s.fscore(truth, predicted)['f1'] == .5
    y = np.zeros((4, 9))
    p = y.copy()
    y[:2, s.MI['CD3']] = y[:2, s.MI['CD19']] = 2
    p[0, s.MI['CD3']] = p[0, s.MI['CD19']] = 2
    gates = dict(positive=[1] * 9, cd45_hi=-1, cd15_lo=1)
    dp = s.compute_metrics(y, p, gates)['double_positive_CD3_CD19']
    assert dp == dict(true_rate=.5, predicted_rate=.25)


def test_knn_revealed_training_labels_only():
    tr = np.array([[0.], [1.], [3.]], np.float32)
    te = np.array([[.1], [2.9]], np.float32)
    y = np.tile(np.array([10., 20., 30.])[:, None], (1, 9))
    mask = np.ones_like(y, bool)
    mask[0, 0] = False
    pred, counts = s.knn_impute(tr, te, y, mask, k=1, query_chunk=1, reference_chunk=1)
    assert pred[0, 0] == 20 and pred[0, 1] == 10
    assert (pred[1] == 30).all()
    y[~mask] = 1e12
    again, _ = s.knn_impute(tr, te, y, mask, k=1, query_chunk=2, reference_chunk=2)
    assert np.array_equal(pred, again)
    pred, counts = s.knn_impute(tr, te, y, mask, k=50)
    assert counts[0]['effective_k'] == 2
    assert pred[0, 0] == 25


def test_gaussian_nll_known_and_gradient():
    mu = torch.zeros(3, 9, requires_grad=True)
    y = torch.ones_like(mu)
    diag = torch.ones_like(mu, requires_grad=True)
    factor = torch.zeros(3, 9, 2, requires_grad=True)
    value = s.gaussian_nll(mu, y, (diag, factor))
    assert value.item() == pytest.approx(4.5 * (1 + math.log(2 * math.pi)))
    value.backward()
    assert torch.isfinite(mu.grad).all() and torch.isfinite(factor.grad).all()


def arguments(pack, out, regime='single', extra=()):
    return ['--pack', str(pack), '--out', str(out), '--test-donor', '0', '--regime', regime,
            '--device', 'cpu', '--workers', '0', '--threads', '1', '--epochs', '2', '--bs', '32',
            '--max-val', '24', '--max-test', '24', '--smoke-tiny-model', '--knn-impute', '3', *extra]


@pytest.mark.parametrize('regime,extra', [('full', ['--cov-rank', '2']), ('single', []),
                                         ('single_pooled', ['--pooled-mode', 'inverse']),
                                         ('single_pooled', ['--pooled-mode', 'head'])])
def test_end_to_end(pack, tmp_path, regime, extra):
    assert s.main(arguments(pack, tmp_path, regime, extra)) == 0
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['status'] == 'complete'
    assert result['setup']['split']['test_donor'] == 0
    assert result['steps'] == 8
    metrics = json.loads((tmp_path / 'metrics.json').read_text())
    assert metrics['n_test'] == 24
    with np.load(tmp_path / 'eval_preds.npz') as pred:
        assert pred['P'].shape == pred['Y'].shape == pred['P_knn'].shape == (24, 9)
        assert (s.donor_ids(pred['sample']) == 0).all()
        assert np.isfinite(pred['P']).all()
        assert ('cov_diag' in pred) == (regime == 'full')


def test_exact_resume_and_reject_changed_recipe(pack, tmp_path):
    whole, resumed = tmp_path / 'whole', tmp_path / 'resumed'
    s.main(arguments(pack, whole))
    s.main(arguments(pack, resumed, extra=['--stop-after-steps', '2']))
    interrupted = torch.load(resumed / 'ckpt.pt', weights_only=False)
    assert interrupted['epoch'] == 0 and interrupted['batch'] == 2
    assert not (resumed / 'eval_preds.npz').exists()
    s.main(arguments(pack, resumed))
    a = torch.load(whole / 'ckpt.pt', weights_only=False)
    b = torch.load(resumed / 'ckpt.pt', weights_only=False)
    assert a['history'] == b['history']
    assert a['scheduler'] == b['scheduler']
    for k in a['model']:
        assert torch.equal(a['model'][k], b['model'][k]), k
    assert np.array_equal(np.load(whole / 'eval_preds.npz')['P'], np.load(resumed / 'eval_preds.npz')['P'])
    assert json.loads((resumed / 'result.json').read_text())['resumed']
    with pytest.raises(ValueError, match='Refusing resume'):
        s.main(arguments(pack, resumed, extra=['--lr', '.001']))
    # Completed checkpoints can be scored again without extra optimiser steps.
    s.main(arguments(pack, resumed))
    assert torch.load(resumed / 'ckpt.pt', weights_only=False)['step'] == 8


@pytest.mark.parametrize('sig', [signal.SIGUSR1, signal.SIGTERM])
def test_signal_checkpoint(pack, tmp_path, monkeypatch, sig):
    original = s.augment
    sent = False
    def sending_augment(x):
        nonlocal sent
        if not sent:
            os.kill(os.getpid(), sig)
            sent = True
        return original(x)
    monkeypatch.setattr(s, 'augment', sending_augment)
    s.main(arguments(pack, tmp_path))
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['status'] == 'interrupted' and result['reason'] == sig.name
    ckpt = torch.load(tmp_path / 'ckpt.pt', weights_only=False)
    assert ckpt['step'] == 1 and ckpt['batch'] == 1
    monkeypatch.setattr(s, 'augment', original)
    s.main(arguments(pack, tmp_path))
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == 'complete'


def test_production_resnet_cpu_forward_backward():
    torch.set_num_threads(1)
    model = s.SandboxModel()
    model.train()
    output = model(torch.zeros(2, 3, 128, 128))
    assert output[0].shape == (2, 9) and output[3].shape == (2, 2048)
    output[0].square().mean().backward()
    assert model.encoder.conv1.weight.grad is not None


def test_augmentation_preserves_channel_alignment():
    x = torch.zeros(12, 3, 128, 128)
    x[:, :, 40:60, 50:70] = torch.tensor([1, 2, 3])[None, :, None, None]
    out = s.augment(x)
    assert torch.equal(out[:, 1], 2 * out[:, 0])
    assert torch.equal(out[:, 2], 3 * out[:, 0])
    assert (out[:, 0].sum((1, 2)) == 400).all()


@pytest.mark.parametrize('steps', [1, 2, 10])
def test_short_onecycle_schedules(pack, tmp_path, steps):
    s.main(arguments(pack, tmp_path, extra=['--epochs', str(steps), '--bs', '256', '--knn-impute', '0']))
    assert torch.load(tmp_path / 'ckpt.pt', weights_only=False)['step'] == steps


def test_resume_at_epoch_boundary(pack, tmp_path):
    s.main(arguments(pack, tmp_path, extra=['--stop-after-steps', '4']))
    ckpt = torch.load(tmp_path / 'ckpt.pt', weights_only=False)
    assert ckpt['epoch'] == 0 and ckpt['batch'] == 4
    s.main(arguments(pack, tmp_path))
    assert torch.load(tmp_path / 'ckpt.pt', weights_only=False)['epoch'] == 2


def test_signal_during_evaluation_keeps_training_checkpoint(pack, tmp_path, monkeypatch):
    original = s.predict
    def interrupted_predict(*args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)
        return original(*args, **kwargs)
    monkeypatch.setattr(s, 'predict', interrupted_predict)
    s.main(arguments(pack, tmp_path))
    result = json.loads((tmp_path / 'result.json').read_text())
    assert result['status'] == 'interrupted' and result['phase'] == 'evaluation'
    ckpt = torch.load(tmp_path / 'ckpt.pt', weights_only=False)
    assert ckpt['epoch'] == 2 and ckpt['step'] == 8
    monkeypatch.setattr(s, 'predict', original)
    s.main(arguments(pack, tmp_path))
    assert json.loads((tmp_path / 'result.json').read_text())['status'] == 'complete'


@pytest.mark.parametrize('regime,extra', [('single', []), ('single_pooled', ['--pooled-mode', 'head']),
                                         ('single_pooled', ['--pooled-mode', 'inverse'])])
def test_hidden_labels_and_test_truth_cannot_change_trained_model(pack, tmp_path, regime, extra):
    with np.load(pack / 'labels.npz') as f:
        altered = {k: f[k].copy() for k in f.files}
    rows = np.arange(len(altered['Y']))
    mask, pmask = s.reveal_masks(rows, regime, 42)
    altered['Y'][~mask] += 1000
    # Pooled raw measurements stay intact. On single-stain cells, all hidden raw values change.
    altered['I_raw'][(~mask) & (~pmask[:, :1])] -= 100000
    test_rows = s.donor_ids(altered['sample']) == 0
    altered['Y'][test_rows] = -500
    altered['I_raw'][test_rows] = -1000
    changed = tmp_path / 'changed_pack'
    changed.mkdir()
    (changed / 'lf.npy').symlink_to(pack / 'lf.npy')
    np.savez(changed / 'labels.npz', **altered)
    original_out, altered_out = tmp_path / 'original', tmp_path / 'altered'
    s.main(arguments(pack, original_out, regime, extra))
    s.main(arguments(changed, altered_out, regime, extra))
    original = torch.load(original_out / 'ckpt.pt', weights_only=False)
    modified = torch.load(altered_out / 'ckpt.pt', weights_only=False)
    assert original['history'] == modified['history']
    for key in original['model']:
        assert torch.equal(original['model'][key], modified['model'][key]), key
    assert np.array_equal(np.load(original_out / 'eval_preds.npz')['P'],
                          np.load(altered_out / 'eval_preds.npz')['P'])
