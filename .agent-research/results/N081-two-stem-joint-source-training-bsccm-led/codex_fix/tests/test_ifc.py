"""CPU mechanics: real N074 LF fixture, synthetic IFC pack and 48-channel SSL weights."""
import ast
import contextlib
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n074_support import load, STAGE, REPO
n074 = load()
n074.TMP = Path('/tmp/n081_fix/ifc_n074_fixture')
n074.TMP.mkdir(parents=True, exist_ok=True)
import numpy as np
import torch
import timm
import bsccm.data as data
import bsccm.gpu_norm as gpu_norm
TMP = Path('/tmp/n081_fix/ifc')
TMP.mkdir(parents=True, exist_ok=True)


def make_ifc(path, order=None):
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(81)
    names = ['CD15', 'Siglec8', 'CD14', 'CD19', 'CD3', 'CD45', 'CD4', 'CD56', 'CD8']
    x = rng.normal(size=(24, 3, 128, 128)).astype('f2')
    y = rng.normal(size=(24, 9)).astype('f4')
    mask = np.ones_like(y, bool)
    # NaNs and explicit missing labels must be inert, including an all-missing marker.
    y[0, 0] = np.nan; mask[1, 1] = False; mask[:16, 8] = False
    sample = np.repeat(['sample0', 'sample1', 'sample2'], 8)
    scale = np.array([2., 3., 4.], 'f4')
    if order is None: order = np.arange(9)
    np.save(path/'lf.npy', x)
    np.savez(path/'labels.npz', Y=y[:, order], names=np.array(names)[order], mask=mask[:, order], sample=sample, chan_scale=scale)
    return x, y, mask, sample, scale


def harness(extra=()):
    return n074.harness(STAGE/'train_lf.py', SOURCE, ssl=SSL,
                        extra=('--ext-ifc', str(IFC),) + tuple(extra))


class IFCTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global SOURCE, SSL, IFC
        SOURCE = n074.make_fixture()
        torch.manual_seed(81)
        net = timm.create_model('resnet50', pretrained=False, in_chans=48, num_classes=0)
        SSL = TMP/'ssl48.pt'; torch.save(net.state_dict(), SSL); del net
        IFC = TMP/'pack'
        cls.fixture = make_ifc(IFC)

    def tearDown(self): gc.collect()

    def test_a_split_normalization_mask_and_stem(self):
        ns = harness()
        pack = ns['ifc']; model = ns['model']
        self.assertEqual(pack.train_donors, ['sample0', 'sample1'])
        np.testing.assert_array_equal(pack.tr, np.arange(16))
        np.testing.assert_array_equal(pack.ev, np.arange(16, 24))
        self.assertFalse(np.intersect1d(pack.tr, pack.ev).size)
        self.assertFalse(pack.mask[0, 0]); self.assertFalse(pack.mask[1, 1])
        self.assertTrue(np.isfinite(pack.y).all())
        x = torch.from_numpy(self.fixture[0][:2].astype('f4'))
        expected = x / torch.tensor(self.fixture[-1])[None, :, None, None]
        self.assertTrue(torch.equal(ns['prep_ifc'](x, False), expected))
        bf = model['enc'].conv1.weight[:, :1]
        self.assertTrue(torch.equal(model['ifc_stem'].weight, bf.expand(-1, 3, -1, -1)))
        self.assertEqual(tuple(model['enc'].conv1.weight.shape), (64, 48, 7, 7))
        self.assertEqual(sum(1 for k in model if k == 'enc'), 1)
        self.assertEqual(tuple(model['ifc_only'].weight.shape), (4, 2048))
        # Per-access assertion prevents mutated samplers from sneaking in the test donor.
        dataset = ns['IFCSet'](pack, True)
        dataset.rows = pack.ev
        with self.assertRaisesRegex(AssertionError, 'donor'): dataset[0]
        split = ns['IFCData'](str(IFC), 'sample2,sample1', 'sample2')
        self.assertEqual(split.train_donors, ['sample1'])
        for train, test in [('sample2', 'sample2'), ('oops', 'sample2'), ('sample0', 'absent')]:
            with self.assertRaises(ValueError): ns['IFCData'](str(IFC), train, test)
        # DataLoader can cycle without ever serving donor 2.
        for _ in range(len(ns['ext_dtr']) + 2):
            *_, g = ns['next_ext_batch']()
            self.assertEqual(len(g), 2)
            self.assertFalse((pack.sample[g.numpy()] == 'sample2').any())
        print('IFC split, per-read boundary, chan_scale, masked NaNs, stem initialization and cycling checked')

    def test_b_shared_gradients_and_shared_rows(self):
        ns = harness(('--init-ssl', str(n074.SOURCE/'runs/n031_ssl_aug_ex0/encoder_final.pt')))
        ns['el'] = 0.0  # N074 extracts the batch body; supply the enclosing epoch accumulator.
        n074.one_batch(ns)
        model = ns['model']; trunk = model['enc'].layer1[0].conv1.weight
        ns['reg'].backward(retain_graph=True)
        self.assertGreater(float(trunk.grad.abs().sum()), 0)
        self.assertGreater(float(model['enc'].conv1.weight.grad.abs().sum()), 0)
        self.assertIsNone(model['ifc_stem'].weight.grad)
        ns['opt'].zero_grad(set_to_none=True)
        ns['ext_loss'].backward(retain_graph=True)
        self.assertGreater(float(trunk.grad.abs().sum()), 0)
        self.assertIsNone(model['enc'].conv1.weight.grad)
        self.assertGreater(float(model['ifc_stem'].weight.grad.abs().sum()), 0)
        selected = ns['IFC_REG_ROWS']
        other = [j for j in range(13) if j not in selected]
        self.assertTrue((model['reg'].weight.grad[selected].abs().sum(1) > 0).all())
        self.assertEqual(float(model['reg'].weight.grad[other].abs().sum()), 0.)
        self.assertGreater(float(model['ifc_affine']['scale'].grad.abs().sum()), 0)
        self.assertGreater(float(model['ifc_affine']['offset'].grad.abs().sum()), 0)
        self.assertGreater(float(model['ifc_only'].weight.grad.abs().sum()), 0)
        ns['opt'].zero_grad(set_to_none=True)
        ns['loss'].backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(torch.equal(ns['loss'], ns['reg'] + ns['a'].w_cls * ns['ce'] + ns['a'].ext_w * ns['ext_loss']))
        print('BSCCM and IFC independently give nonzero gradients to the same layer1; IFC touches exactly five original reg rows')

    def test_c_augmentation_and_masked_loss(self):
        ns = harness()
        x = torch.arange(3 * 128 * 128, dtype=torch.float32).reshape(1, 3, 128, 128)
        scaled = ns['prep_ifc'](x, False)
        # Exercise all rotations/reflections and translation endpoints with channel identity preserved.
        for k in range(4):
            for flip in (0, 1):
                for shift in ((-4, 4), (4, -4)):
                    calls = [torch.tensor([shift]), torch.tensor([k]), torch.tensor([flip])]
                    with mock.patch.object(torch, 'randint', side_effect=calls):
                        got = ns['prep_ifc'](x, True)
                    expected = torch.rot90(scaled, k, (-2, -1))
                    if flip: expected = expected.flip(-1)
                    expected = torch.roll(expected, shift, (-2, -1))
                    self.assertTrue(torch.equal(got, expected))
        p = torch.zeros(2, 9, requires_grad=True); y = torch.ones(2, 9); m = torch.zeros(2, 9, dtype=torch.bool)
        m[0, 0] = True; m[:, 1] = True; y[0, 0] = 2
        self.assertEqual(float(ns['ifc_mse'](p, y, m).detach()), 2.5)
        zero = ns['ifc_mse'](p, y, torch.zeros_like(m)); zero.backward()
        self.assertEqual(float(zero.detach()), 0)
        self.assertTrue(torch.isfinite(p.grad).all())

    def test_d_two_step_full_run_resume_and_eval(self):
        for protocol in ('b1to0', 'b0to1'):
            out = TMP/protocol
            if out.exists(): shutil.rmtree(out)
            data.W = gpu_norm.W = str(SOURCE)
            original_set = gpu_norm.RawSet
            class FixtureRawSet(original_set):
                def __init__(self, rows, labels=None): super().__init__(rows, labels, pack=str(SOURCE/'pack'))
            def run(extra=()):
                argv = ['train_lf.py', '--out', str(out), '--protocol', protocol, '--arch', 'resnet50',
                        '--init-ssl', str(SSL), '--epochs', '1', '--max-train', '4', '--bs', '2',
                        '--workers', '0', '--device', 'cpu', '--targets', 'targets_v5cx_v10.npz',
                        '--ext-ifc', str(IFC), '--seed', '81', *extra]
                path = STAGE/'train_lf.py'; ns = dict(__name__='n081_full', __file__=str(path))
                with mock.patch.object(sys, 'argv', argv), mock.patch.object(gpu_norm, 'RawSet', FixtureRawSet), \
                        mock.patch.dict(sys.modules, {'wandb': None}):
                    exec(compile(path.read_text(), str(path), 'exec'), ns)
                return ns
            ns = run()
            self.assertEqual(ns['step'], 2)
            self.assertTrue(np.isfinite(ns['log'][0]['train_loss']))
            self.assertTrue(np.isfinite(ns['log'][0]['ifc_train_loss']))
            with np.load(out/'eval_ifc.npz') as z:
                self.assertEqual(z['P'].shape, (8, 9)); self.assertTrue(np.isfinite(z['P']).all())
                np.testing.assert_array_equal(z['G'], np.arange(16, 24))
                self.assertTrue((z['sample'] == 'sample2').all())
                self.assertEqual(z['names'].tolist(), ns['ifc'].names)
            result = json.loads((out/'result.json').read_text())
            self.assertEqual(result['final']['ifc']['n_eval'], 8)
            self.assertIn('r/CD19', result['final']['ifc'])
            checkpoint = torch.load(out/'ckpt.pt', map_location='cpu', weights_only=False)
            self.assertIn('ifc_setup', checkpoint['cfg'])
            self.assertEqual(checkpoint['cfg']['ifc_setup']['version'], 2)
            self.assertEqual(len([k for k in checkpoint['m'] if k.endswith('ifc_num_batches_tracked')]), 53)
            for key in ('grad_norm_bsccm', 'grad_norm_ifc_weighted', 'grad_norm_joint'):
                self.assertTrue(np.isfinite(result['final'][key])); self.assertGreater(result['final'][key], 0.)
            self.assertEqual(result['final']['clip_count'], 2)
            self.assertEqual(result['final']['clip_fraction'], 1.)
            first = {k: v.clone() for k, v in checkpoint['m'].items()}
            self.assertEqual(checkpoint['s']['last_epoch'], 2)
            # BSCCM eval cannot execute the new path or read any IFC row.
            with mock.patch.object(ns['IFCSet'], '__getitem__', side_effect=AssertionError('IFC eval leakage')), \
                    mock.patch.object(ns['model']['ifc_stem'], 'forward', side_effect=AssertionError('IFC stem in BSCCM')):
                ns['evaluate'](save=False)
            del ns, checkpoint
            gc.collect()
            ns = run()
            self.assertEqual(ns['ep0'], 1)
            for k, v in first.items(): self.assertTrue(torch.equal(v, ns['model'].state_dict()[k]), k)
            del ns, first; gc.collect()
            with self.assertRaisesRegex(SystemExit, 'refusing to resume'):
                run(('--ext-test-donor', 'sample1'))
            # Config still matches, but missing one domain buffer must fail strict resume.
            damaged = torch.load(out/'ckpt.pt', map_location='cpu', weights_only=False)
            del damaged['m']['enc.bn1.ifc_running_var']
            torch.save(damaged, out/'ckpt.pt')
            del damaged
            with self.assertRaisesRegex(RuntimeError, 'Missing key'):
                run()
            print(f'{protocol}: two optimizer steps, donor-2 eval, unchanged BSCCM eval isolation, strict resume and split-change rejection')

    def test_e_validation_reordered_markers_and_off_options(self):
        with self.assertRaisesRegex(SystemExit, '2'):
            harness(('--ext-w', '-1'))
        for extra in [('--arch', 'resnet18'), ('--init-ssl', ''), ('--head', 'twopart'), ('--panel-head',)]:
            with self.assertRaises(SystemExit): harness(extra)
        reordered = TMP/'reordered'
        make_ifc(reordered, np.arange(8, -1, -1))
        ns = harness(('--ext-ifc', str(reordered), '--dual-zoom', '2'))
        x, *_ = next(iter(ns['ext_dtr']))
        self.assertEqual(tuple(ns['fwd_ifc'](ns['prep_ifc'](x, False)).shape), (2, 9))
        self.assertEqual(ns['IFC_ORDER'], [(ns['IFC_SHARED'] + ns['IFC_ONLY']).index(n) for n in ns['ifc'].names])
        del ns; gc.collect()
        baseline = n074.harness(REPO/'scripts/train_lf.py', SOURCE, ssl=SSL)
        patched = n074.harness(STAGE/'train_lf.py', SOURCE, ssl=SSL,
                              extra=('--ext-w', '3', '--ext-train-donors', 'absent', '--ext-test-donor', 'absent'))
        self.assertEqual(baseline['_cfg'](), patched['_cfg']())
        self.assertEqual(baseline['_run_args'](), patched['_run_args']())
        for k, v in baseline['model'].state_dict().items(): self.assertTrue(torch.equal(v, patched['model'].state_dict()[k]), k)

    def test_f_test_donor_poison_cannot_affect_training(self):
        ns = harness(); ns['el'] = 0.0
        n074.one_batch(ns)
        reference = ns['loss'].detach().clone()
        prediction = ns['ext_p'].detach().clone()
        del ns; gc.collect()
        poisoned = TMP/'poisoned'
        x, y, mask, sample, scale = make_ifc(poisoned)
        x[16:] = np.nan; y[16:] = 1e12
        np.save(poisoned/'lf.npy', x)
        with np.load(poisoned/'labels.npz') as z: names = z['names'].copy()
        np.savez(poisoned/'labels.npz', Y=y, mask=mask, sample=sample, names=names, chan_scale=scale)
        ns = harness(('--ext-ifc', str(poisoned))); ns['el'] = 0.0
        n074.one_batch(ns)
        self.assertTrue(torch.equal(reference, ns['loss'].detach()))
        self.assertTrue(torch.equal(prediction, ns['ext_p'].detach()))
        print('NaN test-donor images and extreme test labels leave the training loss/predictions exactly equal')


if __name__ == '__main__': unittest.main(verbosity=2)
