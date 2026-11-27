"""CPU integration regressions for the staged SSL trainer and the original SIGReg file."""
import ast
from contextlib import ExitStack
import hashlib
import importlib.util
import math
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import timm
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / 'scripts/train_lf.py').is_file())
sys.path.insert(0, str(REPO))
PATCHED = HERE.parent / 'train_ssl_lf.py'
if not PATCHED.exists():
    PATCHED = REPO / 'scripts/train_ssl_lf.py'
spec = importlib.util.spec_from_file_location('n091_trainer', PATCHED)
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)
BASE_REV = '4daacbcb53b830e0f3f9e4dc86c48656365979e7'
BASE_SHA = 'fb2ea1e595a5304102be244ddb7cd9ffe6d46d3b1e85d3b1e4c126bb0d8224b0'
BASE_PATH = REPO / 'scripts/train_ssl_lf.py'


def baseline():
    src = BASE_PATH.read_bytes()
    if hashlib.sha256(src).hexdigest() != BASE_SHA:
        src = subprocess.check_output(['git', '-C', str(REPO), 'show',
                                       BASE_REV + ':scripts/train_ssl_lf.py'])
    assert hashlib.sha256(src).hexdigest() == BASE_SHA
    mod = types.ModuleType('n091_original')
    mod.__file__ = str(BASE_PATH)
    exec(compile(src, str(BASE_PATH), 'exec'), mod.__dict__)
    return mod


def strict_supervised_load(path):
    """Execute the model factory and SSL-init block from the live downstream script."""
    source = REPO / 'scripts/train_lf.py'
    tree = ast.parse(source.read_text())
    factory, = [n for n in tree.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == 'a.arch in _N053']
    load, = [n for n in tree.body if isinstance(n, ast.If) and ast.unparse(n.test) == 'SSL']
    args = types.SimpleNamespace(init_ssl=str(path), arch='resnet50', channels='all', phase=1)
    env = dict(a=args, SSL=True, timm=timm, torch=torch, _N053=())
    exec(compile(ast.Module([factory, load], type_ignores=[]), str(source), 'exec'), env)
    return env['net']


class TinyEncoder(nn.Sequential):
    num_features = 8

    def __init__(self):
        super().__init__(nn.Conv2d(48, 8, 3, stride=2), nn.BatchNorm2d(8),
                         nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())


class CPUTorch:
    """Local redirect of the trainer's CUDA entry points. Optimizer/RNG remain real."""
    cuda = types.SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                                 manual_seed_all=lambda _: None, get_rng_state_all=lambda: [],
                                 set_rng_state_all=lambda _: None)

    def __getattr__(self, name):
        return getattr(torch, name)

    def Generator(self, device='cpu'):
        return torch.Generator(device='cpu')

    def autocast(self, device, **kwargs):
        return torch.autocast('cpu', **kwargs)


def read_ckpt(path):
    return torch.load(path / 'ckpt.pt', map_location='cpu', weights_only=False, mmap=True)


class TrainingFixture(unittest.TestCase):
    def setUp(self):
        root = Path('/tmp/n091_codex')
        root.mkdir(exist_ok=True)
        temp = tempfile.TemporaryDirectory(dir=root)
        self.addCleanup(temp.cleanup)
        self.work = Path(temp.name)
        self.draw = np.arange(2058) % 2
        np.savez(self.work / 'targets_v3.npz', batch=self.draw, split=np.arange(2058) % 3)

    def train(self, mod, name, *, exclude=0, epochs=1, steps=0, objective=None,
              local=0, real=False, per_epoch=3, interrupt=False):
        out = self.work / name
        args = ['trainer', '--arm', 'aug', '--out', str(out), '--bs', '2', '--workers', '0',
                '--seed', '17', '--epochs', str(epochs), '--max-steps', str(steps),
                '--rows', 'all', '--exclude-batch', str(exclude)]
        if objective:
            args.extend(['--objective', objective])
        if local:
            args.extend(['--local-crops', str(local)])
        if objective == 'dino' and not real:
            args.extend(['--prototypes', '19'])
        reads, health_reads, records, losses = [], [], [], []
        size = 128 if real else (80 if local else 16)
        # One deterministic synthetic raw record per global row, independent of access order.
        pixels = torch.arange(24 * size * size).reshape(24, size, size)
        def get_record(ds, i):
            row = int(ds.rows[i])
            assert self.draw[row] != exclude, ('excluded image read', row)
            (reads if isinstance(ds, mod.SSLCells) else health_reads).append(row)
            raw = ((pixels + row * 7) % 97).to(torch.float32) / 97
            return raw[:23], raw[23], row

        rawset = mod.RawSet
        def health_set(rows):
            self.assertEqual(len(rows), 1024)
            self.assertTrue(np.all(self.draw[rows] != exclude))
            return rawset(rows[:4])

        class ShortEpoch(mod.EpochBatches):
            @property
            def total(self):
                return per_epoch

        loss_name = 'dino_loss' if objective == 'dino' else 'ssl_loss'
        original_loss = getattr(mod, loss_name)
        def record_loss(*a, **kw):
            result = original_loss(*a, **kw)
            losses.append(result[0].detach().item())
            if interrupt and len(losses) == 1:
                signal.raise_signal(signal.SIGTERM)
            return result

        run = types.SimpleNamespace(log=lambda r, step: records.append(r), finish=lambda: None)
        fake_wandb = types.SimpleNamespace(init=lambda **kw: run)
        with ExitStack() as ctx:
            def replace(obj, name, value):
                ctx.enter_context(patch.object(obj, name, value))
            ctx.enter_context(patch.dict(sys.modules, wandb=fake_wandb))
            replace(sys, 'argv', args)
            replace(mod, 'W', str(self.work))
            replace(mod, 'torch', CPUTorch())
            replace(torch.Tensor, 'cuda', lambda x, **kw: x)
            replace(nn.Module, 'cuda', lambda x: x)
            replace(mod, 'EpochBatches', ShortEpoch)
            replace(rawset, '__getitem__', get_record)
            replace(mod, 'RawSet', health_set)
            replace(mod, 'GPUNorm', lambda device: lambda raw, ph, *a: torch.cat((raw, ph[:, None]), 1))
            replace(mod, loss_name, record_loss)
            if not real:
                replace(timm, 'create_model', lambda *a, **kw: TinyEncoder())
            old_handler = signal.getsignal(signal.SIGTERM)
            ctx.callback(signal.signal, signal.SIGTERM, old_handler)
            mod.main()
        self.assertGreater(len(reads), 0)
        if records:
            self.assertGreater(len(health_reads), 0)
        return losses, records

    def equal_state(self, left, right):
        self.assertEqual(left.keys(), right.keys())
        for k in left:
            a = left[k].contiguous().reshape(-1).view(torch.uint8)
            b = right[k].contiguous().reshape(-1).view(torch.uint8)
            self.assertTrue(torch.equal(a, b), k)


class LegacyTest(TrainingFixture):
    def test_sigreg_identity_and_old_checkpoint(self):
        old = baseline()
        reference, _ = self.train(old, 'old')
        current, _ = self.train(trainer, 'current')
        first, _ = self.train(old, 'cross_version', steps=1)
        rest, _ = self.train(trainer, 'cross_version', objective='sigreg')
        self.assertEqual(len(reference), 3)
        self.assertEqual(reference, current)
        self.assertEqual(reference, first + rest)
        orig = read_ckpt(self.work / 'old')
        for name in ('current', 'cross_version'):
            actual = read_ckpt(self.work / name)
            self.assertEqual(orig.keys(), actual.keys())
            self.assertEqual(orig['config'], actual['config'])
            self.equal_state(orig['model'], actual['model'])
            self.equal_state(orig['projector'], actual['projector'])
        print('SIGReg baseline/default/legacy-resume losses', reference, flush=True)


class ResumeTest(TrainingFixture):
    def test_dino_resume_locals_and_draws(self):
        kw = dict(objective='dino', local=2, exclude=1, epochs=2, per_epoch=2)
        full, full_logs = self.train(trainer, 'full', **kw)
        first, _ = self.train(trainer, 'resumed', interrupt=True, **kw)
        ck = read_ckpt(self.work / 'resumed')
        scale = 'last.parametrizations.weight.original0'
        self.assertTrue(ck['projector'][scale].eq(1).all())
        self.assertGreater(ck['center'].abs().sum().item(), 0)
        self.assertAlmostEqual(ck['probability_sum'].sum().item(), 2, places=5)
        self.assertEqual(ck['momentum'], 0.996)
        self.assertEqual(ck['opt']['param_groups'][0]['weight_decay'], 0.04)
        self.assertFalse((self.work / 'resumed/encoder_final.pt').exists())
        rest, resume_logs = self.train(trainer, 'resumed', **kw)
        self.assertEqual(len(first), 1)
        self.assertEqual(len(full), 4)
        self.assertEqual(full, first + rest)
        for a, b in zip(full_logs, resume_logs, strict=True):
            for metric in ('loss', 'teacher_entropy', 'teacher_kl_uniform', 'teacher_marginal_kl_uniform'):
                self.assertEqual(a[metric], b[metric])
                self.assertTrue(math.isfinite(a[metric]))
        end = read_ckpt(self.work / 'full')
        resumed = read_ckpt(self.work / 'resumed')
        for key in ('model', 'projector', 'teacher', 'teacher_head'):
            self.equal_state(end[key], resumed[key])
        self.assertTrue(torch.equal(end['center'], resumed['center']))
        self.assertTrue(end['projector'][scale].eq(1).all())   # Claude (red-team N091): DINO norm_last_layer keeps g == 1
        self.assertEqual(end['momentum'], 1.)
        self.assertAlmostEqual(end['opt']['param_groups'][0]['weight_decay'], .4)
        self.assertEqual(end['teacher']['1.num_batches_tracked'].item(), 2 * len(full))
        self.assertEqual(end['model']['1.num_batches_tracked'].item(), 4 * len(full))
        self.assertTrue(np.array_equal(end['rows'], np.flatnonzero(self.draw != 1)))
        print('DINO uninterrupted/resumed next-step loss', full[1], rest[0], flush=True)


class ExportTest(TrainingFixture):
    def test_three_steps_real_resnet_strict_teacher_export(self):
        losses, records = self.train(trainer, 'r50', objective='dino', real=True)
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(map(math.isfinite, losses)))
        self.assertAlmostEqual(records[0]['loss'], np.mean(losses), places=6)
        self.assertAlmostEqual(records[0]['teacher_entropy'] + records[0]['teacher_kl_uniform'],
                               math.log(4096), places=5)
        ck = read_ckpt(self.work / 'r50')
        self.assertTrue(np.array_equal(ck['rows'], np.flatnonzero(self.draw != 0)))
        self.assertFalse(torch.equal(ck['teacher']['conv1.weight'], ck['model']['conv1.weight']))
        net = strict_supervised_load(self.work / 'r50/encoder_final.pt')
        self.assertEqual(net.conv1.in_channels, 48)
        self.equal_state(ck['teacher'], net.state_dict())
        print('ResNet-50 DINO losses', losses, flush=True)


class ObjectiveTest(unittest.TestCase):
    def test_cross_view_loss_and_targets(self):
        rng = torch.Generator().manual_seed(77)
        teachers = [torch.randn(3, 5, generator=rng, requires_grad=True) for _ in range(2)]
        students = [torch.randn(3, 5, generator=rng, requires_grad=True) for _ in range(4)]
        center = torch.rand(5, generator=rng)
        loss, entropy, mean, marginal = trainer.dino_loss(students, teachers, center, .07)
        with torch.no_grad():
            q = [((t - center) / .07).softmax(-1) for t in teachers]
            pairs = [(0, 1), (1, 0), (0, 2), (1, 2), (0, 3), (1, 3)]
            expected = sum(nn.functional.cross_entropy(students[j] / .1, q[i]) for i, j in pairs) / 6
            torch.testing.assert_close(loss, expected)
            torch.testing.assert_close(mean, torch.cat(teachers).mean(0))
            torch.testing.assert_close(marginal, torch.cat(q).mean(0))
            torch.testing.assert_close(entropy, torch.distributions.Categorical(probs=torch.cat(q)).entropy().mean())
        loss.backward()
        self.assertTrue(all(t.grad is None for t in teachers))
        self.assertTrue(all(s.grad.abs().sum() > 0 for s in students))
        self.assertAlmostEqual(trainer.teacher_temperature(0, 40), .04)
        self.assertAlmostEqual(trainer.teacher_temperature(5, 40), .04 + .03 * 5 / 11)
        self.assertAlmostEqual(trainer.teacher_temperature(11, 40), .07)
        self.assertAlmostEqual(trainer.teacher_temperature(39, 40), .07)

    def test_local_geometry_and_ema(self):
        view = torch.arange(80 * 80).reshape(1, 1, 80, 80).expand(6, 48, -1, -1)
        crop = trainer.local_crop(view, torch.Generator().manual_seed(33))
        self.assertEqual(crop.shape, (6, 48, 64, 64))
        self.assertGreater(crop[:, 0, 0, 0].unique().numel(), 1)
        for source, result in zip(view, crop):
            y, x = divmod(result[0, 0, 0].item(), 80)
            self.assertTrue(0 <= y <= 16 and 0 <= x <= 16)
            self.assertTrue(torch.equal(result, source[:, y:y + 64, x:x + 64]))
        student, teacher = nn.Linear(2, 2), nn.Linear(2, 2)
        before = teacher.weight.detach().clone()
        trainer.update_teacher(student, teacher, .996)
        torch.testing.assert_close(teacher.weight, .996 * before + .004 * student.weight)


if __name__ == '__main__':
    unittest.main(verbosity=2)
