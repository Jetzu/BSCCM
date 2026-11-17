"""CPU tests: python -m unittest -v test_phys_aug.py (no training or GPU).

Works both in the N041 delivery directory and after integration in repo/tests.
"""
import ast
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = next(p for p in (HERE, *HERE.parents) if (p / 'bsccm/data.py').exists())
sys.path.insert(0, str(REPO))
MODULE = next(p for p in (HERE / 'bsccm/phys_aug.py', HERE.parent / 'bsccm/phys_aug.py', REPO / 'bsccm/phys_aug.py') if p.exists())
spec = importlib.util.spec_from_file_location('bsccm.phys_aug', MODULE)
phys = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phys)
from bsccm.gpu_norm import GPUNorm

TRAIN = next(p for p in (HERE / 'scripts/train_lf.py', HERE.parent / 'scripts/train_lf.py', REPO / 'scripts/train_lf.py') if p.exists())
TREE = ast.parse(TRAIN.read_text())


def aug(probability=1., **kwargs):
    return phys.PhysAug(probability, .8, [1., 1., 1., 1.], **kwargs)


def definition(name, env):
    node = next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(TRAIN), 'exec'), env)
    return env[name]


class PhysAugTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_zero_is_identity_without_rng_use(self):
        x = torch.randint(0, 4096, (3, 23, 16, 16), dtype=torch.int16)
        gen = torch.Generator().manual_seed(13)
        state = gen.get_state().clone()
        y = aug(0.)(x, generator=gen)
        self.assertIs(x, y)
        self.assertTrue(torch.equal(state, gen.get_state()))
        self.assertEqual(y.dtype, torch.int16)

    def test_shape_dtype_determinism_and_no_mutation(self):
        x = torch.randint(50, 3000, (8, 23, 24, 24), dtype=torch.int16)
        before = x.clone()
        transform = aug(.7)
        y = transform(x, generator=torch.Generator().manual_seed(29))
        z = transform(x, generator=torch.Generator().manual_seed(29))
        self.assertTrue(torch.equal(y, z))
        self.assertTrue(torch.equal(x, before))
        self.assertEqual(x.shape, y.shape)
        self.assertEqual(y.dtype, torch.float32)
        self.assertTrue(torch.isfinite(y).all())
        self.assertGreaterEqual(float(y.min()), 0.)
        self.assertLessEqual(float(y.max()), 4095.)
        self.assertFalse(torch.equal(y, transform(x, generator=torch.Generator().manual_seed(30))))

    def test_gain_offset_probability_and_defocus_statistics(self):
        transform = aug(.7)
        p = transform.sample_parameters(60000, generator=torch.Generator().manual_seed(7))
        self.assertAlmostEqual(float(p['active'].float().mean()), .7, delta=.008)
        log_gain = p['gain'].log()
        self.assertAlmostEqual(float(log_gain.mean()), 0., delta=.001)
        self.assertAlmostEqual(float(log_gain.std()), .1, delta=.001)
        self.assertLess(float(torch.corrcoef(log_gain.T)[0, 1].abs()), .02)
        self.assertAlmostEqual(float(p['offset'].std()), 1., delta=.01)
        self.assertAlmostEqual(float(p['z_um'].mean()), 0., delta=.01)
        self.assertAlmostEqual(float(p['z_um'].std()), 1 / np.sqrt(3), delta=.01)
        self.assertLessEqual(float(p['z_um'].abs().max()), 1.)
        s = p['shift_xy']
        self.assertTrue(torch.equal(s[:, 19], -s[:, 20]))
        self.assertTrue(torch.equal(s[:, 18], -s[:, 21]))
        self.assertGreater(float(s[:, 14, 1].std()), float(s[:, 2, 1].std()))
        self.assertAlmostEqual(float(s[:, 14, 1].std()), 3.1475291164 / np.sqrt(3), delta=.02)
        self.assertTrue(torch.equal(s[:, [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]], torch.zeros_like(s[:, [0, 1, 4, 5, 8, 9, 12, 13, 16, 17]])))

    def test_family_gains_and_offsets_shared_in_actual_output(self):
        transform = aug(1., defocus_um=0., shot_scale=0.)
        seed = 33
        p = transform.sample_parameters(10, generator=torch.Generator().manual_seed(seed))
        x = torch.full((10, 23, 8, 8), 1030, dtype=torch.int16)
        y = transform(x, generator=torch.Generator().manual_seed(seed))
        expected = 30 + 1000 * p['gain'][:, transform.families] + p['offset'][:, transform.families]
        torch.testing.assert_close(y, expected[:, :, None, None].expand_as(y))
        self.assertEqual(phys.FAMILIES[22], phys.FAMILIES[0])

    def test_no_channel_mixing_and_fractional_shift_geometry(self):
        transform = phys.PhysAug(1., .8, [0.] * 4, gain_log_sd=0., shot_scale=0.)
        seed = 17
        p = transform.sample_parameters(12, generator=torch.Generator().manual_seed(seed))
        x = torch.full((12, 23, 32, 32), 30, dtype=torch.int16)
        # Only one channel is nonconstant: any other change implies mixing.
        x[:, 14, 16, 16] = 1030
        y = transform(x, generator=torch.Generator().manual_seed(seed))
        other = [c for c in range(23) if c != 14]
        self.assertTrue(torch.equal(y[:, other], x[:, other].float()))
        mass = y[:, 14] - 30
        coord = torch.arange(32, dtype=torch.float32)
        cx = (mass.sum(1) * coord).sum(1) / mass.sum((1, 2))
        cy = (mass.sum(2) * coord).sum(1) / mass.sum((1, 2))
        torch.testing.assert_close(cx - 16, p['shift_xy'][:, 14, 0], atol=3e-6, rtol=0)
        torch.testing.assert_close(cy - 16, p['shift_xy'][:, 14, 1], atol=3e-6, rtol=0)
        self.assertTrue(((cy - cy.round()).abs() > .02).any())

    def test_unperturbed_fraction_exact_including_suboffset_counts(self):
        transform = aug(.7)
        seed = 37
        p = transform.sample_parameters(100, generator=torch.Generator().manual_seed(seed))
        x = torch.randint(0, 4096, (100, 23, 8, 8), dtype=torch.int16)
        y = transform(x, generator=torch.Generator().manual_seed(seed))
        self.assertTrue(torch.equal(y[~p['active']], x[~p['active']].float()))
        # No hidden cleaning of sub-offset counts in an otherwise neutral transform.
        neutral = phys.PhysAug(1., .8, [0.] * 4, gain_log_sd=0., defocus_um=0., shot_scale=0.)
        self.assertTrue(torch.equal(neutral(x), x.float()))

    def test_added_shot_variance_matches_signal(self):
        transform = phys.PhysAug(1., .8, [0.] * 4, gain_log_sd=0., defocus_um=0.)
        x = torch.full((64, 23, 24, 24), 130, dtype=torch.int16)
        x[32:] = 1030
        y = transform(x, generator=torch.Generator().manual_seed(19)) - x.float()
        for part, signal in ((y[:32], 100.), (y[32:], 1000.)):
            self.assertAlmostEqual(float(part.mean()), 0., delta=.15)
            self.assertAlmostEqual(float(part.var()), .8 * signal, delta=.02 * .8 * signal)

    def test_calibration_recovers_synthetic_gain_and_validates_before_image_read(self):
        rng = np.random.default_rng(4)
        n, h = 160, 32
        signal = rng.uniform(100, 2500, (n, 23, 1, 1))
        x = (30 + signal + rng.normal(size=(n, 23, h, h)) * np.sqrt(.8 * signal + 9)).astype(np.float32)
        batch, split, hist = np.zeros(n, int), np.zeros(n, int), np.full(n, -1)
        class SourceOnly:
            shape = x.shape
            def __init__(self): self.read = []
            def __getitem__(self, row): self.read.append(row); return x[row]
        led = SourceOnly()
        c = phys.calibrate_source_borders(led, np.arange(128), batch, split, hist, max_rows=128, border=4)
        self.assertAlmostEqual(c['gain_adu_per_e'], .8, delta=.08)
        self.assertEqual(led.read, c['rows'])
        self.assertTrue(set(led.read).issubset(range(128)))
        for bad in ('draw', 'split', 'gs'):
            bb, ss, hh = batch.copy(), split.copy(), hist.copy()
            if bad == 'draw': bb[0] = 1
            if bad == 'split': ss[0] = 2
            if bad == 'gs': hh[0] = 0
            led.read.clear()
            with self.assertRaises(ValueError): phys.calibrate_source_borders(led, np.arange(128), bb, ss, hh, border=4)
            self.assertEqual(led.read, [])

    def test_prep_eval_and_off_path_match_original_norm_and_preserve_phase(self):
        # Real GPUNorm maths on a small synthetic CPU fixture, CUDA transfers stubbed.
        norm = GPUNorm.__new__(GPUNorm)
        norm.py = norm.px = torch.tensor([64, 64])
        norm.slide = torch.tensor([0, 0]); norm.scale = torch.ones(1, 23)
        norm.s_c = torch.ones(23); norm.bg = torch.full((23, 128, 128), 1000.)
        norm.H = 128; norm.ar = torch.arange(128)
        env = dict(a=SimpleNamespace(phase=1, lowpass=0., zoom=1), gnorm=norm, CH=list(range(23)),
                   phys_aug=None, torch=torch, np=np, F=F)
        prep = definition('prep', env)
        x = torch.randint(100, 2500, (2, 23, 128, 128), dtype=torch.int16)
        ph = torch.randn(2, 128, 128); rows = torch.arange(2)
        with patch.object(torch.Tensor, 'cuda', lambda tensor, **kw: tensor):
            for train in (False, True):
                torch.manual_seed(42); expected = norm(x, ph, rows, list(range(23)), True, shift=4 if train else 0)
                torch.manual_seed(42); got = prep(x, ph, rows, train)
                self.assertTrue(torch.equal(got, expected))
            env['phys_aug'] = aug(1.)
            self.assertTrue(torch.equal(prep(x, ph, rows, False), norm(x, ph, rows, list(range(23)), True)))
            # Both training views still pass the identical measured phase to GPUNorm.
            phase_seen = []
            def record(raw, phase, *args, **kwargs):
                phase_seen.append(phase.clone())
                return norm(raw, phase, *args, **kwargs)
            env['gnorm'] = record
            prep(x, ph, rows, True); prep(x, ph, rows, True)
            self.assertTrue(all(torch.equal(v, ph) for v in phase_seen))

    def test_checkpoint_config_defaults_omit_new_options(self):
        defaults = next(n for n in TREE.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_NEW_DEFAULTS' for t in n.targets))
        env = dict(a=SimpleNamespace(out='tmp', workers=1, adabn=0, arch='resnet18', phys_aug=0., consist=0.), PHYS=False)
        exec(compile(ast.Module(body=[defaults], type_ignores=[]), str(TRAIN), 'exec'), env)
        cfg = definition('_cfg', env)
        self.assertEqual(cfg(), {'arch': 'resnet18'})
        env['PHYS'] = True; env['a'].phys_aug = .7; env['a'].consist = .1; env['phys_calibration'] = {'source_draw': 1}
        self.assertEqual(cfg()['phys_aug'], .7)
        self.assertEqual(cfg()['consist'], .1)
        self.assertEqual(cfg()['phys_calibration'], {'source_draw': 1})

    def test_actual_consistency_loss_has_two_symmetric_gradients(self):
        # Execute the exact new training loss block, without importing the CLI.
        source = TRAIN.read_text()
        start = source.index('        consistency = p.new_zeros(())')
        end = source.index('        opt.zero_grad', start)
        import textwrap
        block = compile(textwrap.dedent(source[start:end]), str(TRAIN), 'exec')
        for weight in (0., .1):
            p = torch.tensor([[1., 2.]], requires_grad=True)
            p2 = torch.tensor([[3., 5.]], requires_grad=True)
            calls = []
            def forward(x): calls.append(x); return p2, None, None, None
            env = dict(p=p, reg=p.sum()*0, ce=p.sum()*0, a=SimpleNamespace(consist=weight, w_cls=.5),
                       raw=object(), ph=object(), g=object(), prep=lambda *args: args[0], fwd=forward, F=F)
            exec(block, env); env['loss'].backward()
            self.assertEqual(len(calls), int(weight > 0))
            if weight:
                torch.testing.assert_close(p.grad, -p2.grad)
                self.assertGreater(float(p.grad.abs().sum()), 0.)
                self.assertAlmostEqual(float(env['consistency'].detach()), 6.5)
            else:
                self.assertIsNone(p2.grad)


if __name__ == '__main__':
    unittest.main()
