"""CPU tests and tiny full-trainer smoke; all temporary files stay under /tmp/codex_n049.

Run from the delivery directory:
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    /tmp/codex_n049/venv/bin/python -m unittest discover -s tests -v
Override BSCCM_REPO if the read-only baseline checkout is elsewhere.
"""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
REPO = Path(os.environ.get('BSCCM_REPO', '/home/hpc/v181bb/v181bb10/BSCCM'))
SCRATCH = Path('/tmp/codex_n049')
SCRATCH.mkdir(exist_ok=True)
sys.path.insert(0, str(REPO))
import bsccm
bsccm.__path__.append(str(ROOT / 'bsccm'))
from bsccm.n049_pospos import load_pospos, masked_soft_bce, source_digest
spec = importlib.util.spec_from_file_location('n049_mixture', ROOT / 'scripts/n049_noise_mixture.py')
mixture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixture)
torch.set_num_threads(1)


def labels(n_source=400, n_target=32):
    rng = np.random.default_rng(941)
    n = n_source + n_target
    names = np.array(['single:' + x for x in ('CD45', 'CD123', 'CD19', 'CD56', 'CD14', 'CD16', 'HLA-DR', 'CD3')]
                     + ['grp:CD123/HLA-DR/CD14', 'grp:CD3/CD19/CD56', 'grp:CD45', 'grp:CD16', 'autofl'])
    m = np.zeros((n, 13), bool)
    m[np.arange(n), np.arange(n) % 8] = True
    m[-n_target:, 0] = True
    y = rng.normal(-1, .3, (n, 13)) + rng.binomial(1, .35, (n, 13)) * 2.5
    y[~m] = 0
    split = np.zeros(n, np.int64)
    split[n_source - 32:] = 2
    hist = np.full(n, -1, np.int64); hist[n_source - 1] = 0
    return dict(Y=y.astype(np.float32), mask=m, names=names, split=split,
                batch=np.r_[np.zeros(n_source, np.int64), np.ones(n_target, np.int64)],
                hist_row=hist, cls3=np.full(n, -1), cls10=np.full(n, -1))


def archive(path, t):
    g = np.flatnonzero((t['batch'] == 0) & (t['hist_row'] < 0))
    m = t['mask'][g, :8]
    meta = dict(schema='n049-v1', source_draw=0, source_sha256=source_digest(t, g, np.arange(8)))
    p = np.where(m, 1 / (1 + np.exp(-t['Y'][g, :8])), 0).astype(np.float32)
    np.savez(path, G=g, P=p, mask=m, fit_mask=m & (t['split'][g, None] == 0),
             names=t['names'][:8], metadata=np.array(json.dumps(meta)))


def mutate(path, **changes):
    with np.load(path) as z: d = dict(z)
    d.update(changes); np.savez(path, **d)


class NoiseTests(unittest.TestCase):
    def test_posterior_matches_latent_enumeration_and_is_soft(self):
        p = dict(pi=.3, eta=.8, phi=.04, mu=[-1., 2.], variance=[.5, .8])
        y = np.linspace(-5, 6, 61)
        density = np.exp(mixture._log_components(y, np.array(p['mu']), np.array(p['variance'])))
        num = p['pi'] * ((1 - p['eta']) * density[:, 0] + p['eta'] * density[:, 1])
        den = num + (1 - p['pi']) * ((1 - p['phi']) * density[:, 0] + p['phi'] * density[:, 1])
        np.testing.assert_allclose(mixture.posterior(y, p), num / den, atol=1e-14)
        self.assertTrue(((mixture.posterior(y, p) > 0) & (mixture.posterior(y, p) < 1)).all())

    def test_em_separated_and_single_gaussian_null(self):
        rng = np.random.default_rng(51)
        y = rng.normal(0, .35, 3000) + rng.binomial(1, .3, 3000) * 3
        fit = mixture.fit_noise_mixture(y)
        self.assertGreater(fit['d_prime'], 6)
        self.assertGreater(fit['delta_bic_1_minus_2'], 1000)
        self.assertGreaterEqual(fit['min_objective_step'], -1e-7)
        self.assertTrue(fit['converged'])
        self.assertEqual(fit['variance'][0], fit['variance'][1])
        grid = np.linspace(y.min() - 3, y.max() + 3, 1000)
        self.assertTrue((np.diff(mixture.posterior(grid, fit)) >= -1e-14).all())
        self.assertTrue(.6 <= fit['eta'] <= .99 and .001 <= fit['phi'] <= .15)
        null = mixture.fit_noise_mixture(rng.normal(size=3000))
        self.assertLess(null['delta_bic_1_minus_2'], 0)
        with self.assertRaises(ValueError): mixture.fit_noise_mixture(np.ones(100))

    def test_fitting_never_uses_other_draw_nonzero_splits_giemsa_or_missing(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
            d = Path(d); t = labels(); np.savez(d / 't.npz', **t)
            a = mixture.generate(d / 't.npz', 0, d / 'a.npz')
            changed = copy.deepcopy(t)
            excluded = (t['batch'] != 0) | (t['split'] != 0) | (t['hist_row'] >= 0)
            changed['Y'][excluded] += 1000
            changed['Y'][~changed['mask']] = -12345
            np.savez(d / 't2.npz', **changed)
            b = mixture.generate(d / 't2.npz', 0, d / 'b.npz')
            self.assertEqual(a['fits'], b['fits'])
            with np.load(d / 'a.npz') as z:
                self.assertTrue((z['P'][~z['mask']] == 0).all())
                self.assertTrue((t['hist_row'][z['G']] < 0).all())
                self.assertTrue((t['batch'][z['G']] == 0).all())
                self.assertEqual(int(z['mask'].sum()), int(t['mask'][z['G'], :8].sum()))


class SupervisionTests(unittest.TestCase):
    def test_soft_bce_value_gradient_and_missing_entries(self):
        x = torch.tensor([[.2, float('nan')], [-.6, .8], [float('nan'), 1.2]], requires_grad=True)
        y = torch.tensor([[.25, float('nan')], [.8, .6], [float('nan'), .1]])
        m = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.bool)
        got = masked_soft_bce(x, y, m)
        expected = (F.binary_cross_entropy_with_logits(x[:2, 0], y[:2, 0])
                    + F.binary_cross_entropy_with_logits(x[1:, 1], y[1:, 1])) / 2
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        got.backward()
        self.assertTrue(torch.equal(x.grad[~m], torch.zeros(2)))
        torch.testing.assert_close(x.grad[m], (x.detach().sigmoid() - y)[m] / 4)
        x2 = torch.full((2, 3), float('nan'), requires_grad=True)
        zero = masked_soft_bce(x2, x2.detach(), torch.zeros_like(x2, dtype=torch.bool))
        self.assertEqual(float(zero.detach()), 0)
        zero.backward(); self.assertTrue(torch.equal(x2.grad, torch.zeros_like(x2)))
        # Unequal counts must weight markers equally rather than label entries equally.
        xx = torch.tensor([[0., 1.], [2., 3.]])
        yy = torch.tensor([[.2, .4], [.6, .8]])
        mm = torch.tensor([[1, 0], [1, 1]], dtype=torch.bool)
        torch.testing.assert_close(masked_soft_bce(xx, yy, mm),
            (F.binary_cross_entropy_with_logits(xx[:, 0], yy[:, 0])
             + F.binary_cross_entropy_with_logits(xx[1:, 1], yy[1:, 1])) / 2)

    def test_archive_integrity_and_content_fingerprint(self):
        with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
            p = Path(d) / 'p.npz'; t = labels(); archive(p, t)
            tr = np.flatnonzero((t['batch'] == 0) & (t['hist_row'] < 0))
            py, pm, ident = load_pospos(p, t, tr, 'b0to1')
            self.assertEqual(pm.sum(), t['mask'][tr, :8].sum())
            self.assertFalse(pm[t['batch'] == 1].any())
            with np.load(p) as z: pp = z['P'].copy()
            pp[0, 0] = .123; mutate(p, P=pp)
            self.assertNotEqual(ident['sha256'], load_pospos(p, t, tr, 'b0to1')[2]['sha256'])
            changed = copy.deepcopy(t); changed['Y'][0, 0] += 1
            with self.assertRaisesRegex(ValueError, 'source labels'): load_pospos(p, changed, tr, 'b0to1')
            for mode in ('b1to0', 'fov'):
                with self.assertRaises(ValueError): load_pospos(p, t, tr, mode)
            with self.assertRaises(ValueError): load_pospos(p, t, tr, 'b0to1', 1)
            for key, value in [('P', np.full_like(pp, np.nan)), ('mask', pm[tr].astype(np.int64)),
                               ('G', tr[::-1]), ('names', t['names'][1:9]),
                               ('fit_mask', np.zeros_like(pm[tr]))]:
                archive(p, t); mutate(p, **{key: value})
                with self.subTest(key=key), self.assertRaises(ValueError): load_pospos(p, t, tr, 'b0to1')
            archive(p, t)
            with np.load(p) as z: pp = z['P'].copy()
            pp[0, 1] = .3; mutate(p, P=pp)
            with self.assertRaisesRegex(ValueError, 'missing'): load_pospos(p, t, tr, 'b0to1')


# Run the real trainer/ResNet/normalisation/RawSet, using a tiny synthetic raw-LF pack.
# Only dataset location and optional baseline CPU adaptation are patched in the harness.
def run_trainer(script, data, out, extra=(), baseline=False):
    import bsccm.data as bd
    import bsccm.gpu_norm as bg
    import timm  # preload before torch shims
    old_rawset = bg.RawSet
    class TinyRawSet(old_rawset):
        def __init__(self, rows, labels=None): super().__init__(rows, labels, pack=str(data / 'pack'))
    args = [str(script), '--out', str(out), '--targets', 'targets_v5.npz', '--protocol', 'b0to1',
            '--channels', 'bf', '--phase', '0', '--epochs', '1', '--bs', '4', '--workers', '0',
            '--max-train', '8', '--seed', '71', *extra]
    if not baseline: args += ['--device', 'cpu']
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(bd, 'W', str(data)))
    stack.enter_context(patch.object(bg, 'W', str(data)))
    stack.enter_context(patch.object(bg, 'RawSet', TinyRawSet))
    stack.enter_context(patch.object(sys, 'argv', args))
    stack.enter_context(patch.dict(sys.modules, {'wandb': None}))
    if baseline:
        # Original is CUDA-only. Map its existing calls to CPU, without changing maths.
        original_tensor, original_load, original_dl = torch.tensor, torch.load, torch.utils.data.DataLoader
        original_norm = bg.GPUNorm
        def tensor_cpu(*a, **kw):
            if kw.get('device') == 'cuda': kw['device'] = 'cpu'
            return original_tensor(*a, **kw)
        def load_cpu(*a, **kw):
            if kw.get('map_location') == 'cuda': kw['map_location'] = 'cpu'
            return original_load(*a, **kw)
        def loader_cpu(*a, **kw):
            if not kw.get('num_workers'):
                kw.pop('persistent_workers', None); kw.pop('prefetch_factor', None)
            kw['pin_memory'] = False
            return original_dl(*a, **kw)
        stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda self, *a, **kw: self))
        stack.enter_context(patch.object(torch.nn.Module, 'cuda', lambda self, *a, **kw: self))
        stack.enter_context(patch.object(torch, 'tensor', tensor_cpu))
        stack.enter_context(patch.object(torch, 'load', load_cpu))
        stack.enter_context(patch.object(torch.utils.data, 'DataLoader', loader_cpu))
        stack.enter_context(patch.object(torch, 'autocast', lambda *a, **kw: contextlib.nullcontext()))
        stack.enter_context(patch.object(bg, 'GPUNorm', lambda *a: original_norm('cpu')))
    try:
        with contextlib.redirect_stdout(io.StringIO()): return runpy.run_path(str(script), run_name='__main__')
    finally: stack.close()


def make_pack(data, t):
    n = len(t['Y']); (data / 'pack').mkdir(parents=True)
    rng = np.random.default_rng(811)
    np.save(data / 'pack/led_array.npy', rng.integers(100, 900, (n, 1, 128, 128), dtype=np.uint16))
    np.save(data / 'pack/dpc.npy', np.zeros((n, 128, 128), np.float32))
    np.save(data / 'backgrounds_p50.npy', np.full((1, 128, 128), 500, np.float32))
    np.savez(data / 'cellmeta.npz', pos_y=np.full(n, 64), pos_x=np.full(n, 64), slide=np.zeros(n, np.int64))
    np.savez(data / 'led_norm.npz', slide_scale=np.ones((1, 1), np.float32), s_c=np.ones(1, np.float32))
    np.savez(data / 'targets_v5.npz', **t)


def assert_nested_equal(test, a, b):
    if isinstance(a, torch.Tensor): test.assertTrue(torch.equal(a, b))
    elif isinstance(a, np.ndarray): np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        test.assertEqual(a.keys(), b.keys())
        for k in a: assert_nested_equal(test, a[k], b[k])
    elif isinstance(a, (list, tuple)):
        test.assertEqual(len(a), len(b))
        for x, y in zip(a, b): assert_nested_equal(test, x, y)
    else: test.assertEqual(a, b)


class TrainerSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(dir=SCRATCH)
        cls.d = Path(cls.tmp.name)
        cls.t = labels(n_source=64, n_target=24)
        make_pack(cls.d / 'data', cls.t)
        archive(cls.d / 'pos.npz', cls.t)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_default_full_cpu_training_bit_identical_and_old_checkpoint_resume(self):
        data = self.d / 'data'; old = self.d / 'old'; new = self.d / 'new'
        run_trainer(REPO / 'scripts/train_lf.py', data, old, baseline=True)
        g = run_trainer(ROOT / 'scripts/train_lf.py', data, new)
        a = torch.load(old / 'ckpt.pt', weights_only=False)
        b = torch.load(new / 'ckpt.pt', weights_only=False)
        for k in ('m', 'o', 's', 'tr', 'ev'): assert_nested_equal(self, a[k], b[k])
        self.assertEqual(a['log'][0]['train_loss'], b['log'][0]['train_loss'])
        with np.load(old / 'eval_preds.npz') as x, np.load(new / 'eval_preds.npz') as y:
            self.assertEqual(x.files, y.files)
            for k in x.files: np.testing.assert_array_equal(x[k], y[k])
        # GPU default config omits ALL new defaults, matching a real old checkpoint.
        g['a'].device = 'cuda'
        self.assertEqual(g['_cfg'](), a['cfg'])
        g['a'].device = 'cpu'
        # CPU is intentionally fingerprinted: convert just this old checkpoint's device
        # field, then exercise the actual resume branch and strict model/optim loading.
        a['cfg']['device'] = 'cpu'; torch.save(a, old / 'ckpt.pt')
        resumed = run_trainer(ROOT / 'scripts/train_lf.py', data, old)
        self.assertEqual(resumed['ep0'], 1)
        self.assertFalse(any(k.startswith('pos.') for k in resumed['model'].state_dict()))

    def test_opt_in_full_cpu_smoke_predictions_loss_and_resume_rejection(self):
        out = self.d / 'enabled'; data = self.d / 'data'; p = self.d / 'pos.npz'
        args = ['--pospos', str(p), '--pos-w', '.25']
        g = run_trainer(ROOT / 'scripts/train_lf.py', data, out, args)
        ck = torch.load(out / 'ckpt.pt', weights_only=False)
        self.assertIn('pos.weight', ck['m']); self.assertTrue(np.isfinite(ck['log'][0]['train_loss']))
        self.assertIn('pospos_id', ck['cfg'])
        self.assertGreater(float(g['model']['pos'].weight.grad.abs().sum()), 0)
        with np.load(out / 'eval_preds.npz') as z:
            self.assertEqual(z['P_pos'].shape, (24, 8)); self.assertEqual(z['P'].shape, (24, 13))
            self.assertTrue(((z['P_pos'] > 0) & (z['P_pos'] < 1)).all())
        resumed = run_trainer(ROOT / 'scripts/train_lf.py', data, out, args)
        self.assertEqual(resumed['ep0'], 1)
        with np.load(p) as z: pp = z['P'].copy()
        pp[0, 0] = .456; mutate(p, P=pp)
        with self.assertRaisesRegex(SystemExit, 'refusing to resume'):
            run_trainer(ROOT / 'scripts/train_lf.py', data, out, args)
        archive(p, self.t)
        with self.assertRaisesRegex(SystemExit, 'refusing to resume'):
            run_trainer(ROOT / 'scripts/train_lf.py', data, out, ['--pospos', str(p), '--pos-w', '.3'])
        # Full trainer also supports pseudo labels + pos labels in the same RawSet tuple.
        tr = np.flatnonzero((self.t['batch'] == 0) & (self.t['hist_row'] < 0))
        teacher = self.d / 'teacher.npz'
        np.savez(teacher, G=tr, P=np.zeros((len(tr), 13), np.float32), names=self.t['names'])
        run_trainer(ROOT / 'scripts/train_lf.py', data, self.d / 'combined',
                    [*args, '--pseudo', str(teacher), '--lowpass', '.5'])
        # Record only compact evidence; never retain temporary raw data or models.
        evidence = dict(cpu=True, architecture='resnet18', input='synthetic BF only, actual RawSet + GPUNorm',
                        train_steps=2, batch_size=4, eval_rows=24,
                        enabled_train_loss=ck['log'][0]['train_loss'], output_shapes={'P': [24, 13], 'P_pos': [24, 8]},
                        default_bit_identical='model, optimizer, scheduler, predictions, training loss',
                        resume='defaults compatible; content and weight changes rejected', torch=torch.__version__)
        (ROOT / 'diagnostics/cpu_smoke.json').write_text(json.dumps(evidence, indent=2) + '\n')


if __name__ == '__main__': unittest.main()
