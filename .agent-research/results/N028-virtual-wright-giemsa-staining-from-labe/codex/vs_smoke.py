"""Bounded CPU component/control smoke test: eight real cells, no training run."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import gc
import json
import multiprocessing as mp
import os
import signal
import time
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from vs_common import (W, VERSION, UNet, PatchGAN, Dino, Pairs, GPUNorm, EpochBatches,
    atomic_npz, atomic_torch, digest, image_metrics, make_loader, normalize_batch,
    read_registration, registered_rgb, restore_rng, retrieval, rng_state, split_rows, translate_pair)
from vs_register import init_worker, fit_one, register_one, phase_shift, ncc
from vs_train import train_step


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tmp', required=True)
    p.add_argument('--threads', type=int, default=48)
    a = p.parse_args()
    tmp = Path(a.tmp); tmp.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.manual_seed(7); rng = np.random.default_rng(7)
    t0 = time.monotonic()
    # Independent sign control: shift a synthetic image, recover inverse translation.
    base = gaussian_filter(rng.normal(size=(128, 128)), 2).astype(np.float32)
    shifted = np.roll(base, (3, -4), axis=(0, 1))
    dy, dx, peak, _ = phase_shift(base, shifted)
    assert (dy, dx) == (-3, 4), (dy, dx, peak)
    assert phase_shift(np.ones((128, 128)), np.ones((128, 128)))[2] == -1
    # Coordinate ramp independently checks output->source and 128/256 shift scaling.
    ramp = np.broadcast_to(np.arange(398, dtype=np.float32)[:, None], (398, 398))
    for size in [128, 256]:
        a0 = registered_rgb(ramp, .7, 0, 0, size)
        a1 = registered_rgb(ramp, .7, 3, 0, size)
        assert np.allclose((a0-a1)[20:-20, 20:-20], 3*398*.7/128, atol=1e-4)
    hc = np.load(W+'/pack/hist_cells.npy')
    targets = W+'/targets_v6.npz'
    tr, te = split_rows(targets, hc, 'fov')
    rows = np.r_[rng.choice(tr, 4, replace=False), rng.choice(te, 4, replace=False)]
    init_worker(W+'/pack')
    scales = np.linspace(.6, .8, 5)
    fit = [fit_one((int(r), scales, 12)) for r in rows[:4]]
    scale = float(scales[np.median([f[0] for f in fit], axis=0).argmax()])
    rgb_scale = float(np.percentile(np.concatenate([f[1] for f in fit]), 99.9))
    tasks = [(int(r), scale, 12) for r in rows]
    res = np.array([register_one(t) for t in tasks])
    # Exercise actual spawn/pickling/memmap initialization, using one NUMA fraction.
    with mp.get_context('spawn').Pool(48, initializer=init_worker, initargs=(W+'/pack',)) as pool:
        parallel = np.array(pool.map(register_one, tasks, chunksize=1))
    assert np.allclose(res, parallel)

    reg = dict(gidx=hc, hist_row=np.arange(len(hc)), scale=np.full(len(hc), scale),
               dy=np.zeros(len(hc), np.int16), dx=np.zeros(len(hc), np.int16),
               peak=np.zeros(len(hc)), good=np.zeros(len(hc), bool), rgb_scale=rgb_scale)
    reg['dy'][rows], reg['dx'][rows], reg['peak'][rows] = res[:, 0], res[:, 1], res[:, 2]
    # Smoke tests keep all eight cells; this is not a production quality-filter decision.
    reg['good'][rows] = True
    reg['meta'] = json.dumps(dict(version=VERSION, holdout='fov', targets_sha256=digest(targets),
                                 report=dict(smoke_only=True, n_good=8)))
    regpath = tmp/'reg.npz'; atomic_npz(regpath, **reg)
    reg, tr, te = read_registration(regpath, targets, W+'/pack', 'fov')
    assert len(tr) == len(te) == 4
    try:
        read_registration(regpath, targets, W+'/pack', 'batch1')
    except ValueError:
        pass
    else:
        raise AssertionError('holdout mismatch accepted')
    norm = GPUNorm('cpu')
    ds = Pairs(reg, tr, 128, W+'/pack')
    batch = next(iter(make_loader(ds, 2, 0)))
    x, y = normalize_batch(batch, norm, 'cpu')
    assert x.shape == (2, 24, 128, 128) and torch.isfinite(x).all()
    x_aug, y_aug = translate_pair(x, y, 4)
    assert x_aug.shape == x.shape and y_aug.shape == y.shape
    sampler = EpochBatches(7, 3, 3)
    full = list(sampler); sampler.start = 1
    assert list(sampler) == full[1:] and sorted(sum(full, [])) == list(range(7))
    state = rng_state(); r = torch.rand(3); restore_rng(state); assert torch.equal(torch.rand(3), r)
    print(json.dumps(dict(stage='geometry_and_data', actual_cells=8, scale=scale,
         median_ncc=float(np.median(res[:, 2])), shifts=res[:, :2].tolist())), flush=True)
    net = UNet(8).to(memory_format=torch.channels_last)
    dino = Dino('S', W+'/models/DinoBloom-S.pth')
    opt = torch.optim.AdamW(net.parameters(), lr=1e-4)
    losses = train_step(net, None, dino, opt, None, x_aug, y_aug)
    assert all(np.isfinite(list(losses.values())))
    assert all(p.grad is None for p in dino.parameters())
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in net.parameters())
    # Exercise conditional GAN gradients without another Dino backward.
    disc = PatchGAN(8)
    dopt = torch.optim.AdamW(disc.parameters(), lr=1e-4)
    gan_losses = train_step(net, disc, None, opt, dopt, x, y, perceptual_weight=0)
    assert all(np.isfinite(list(gan_losses.values())))
    with torch.no_grad():
        net256 = UNet(8, 256)
        assert net256(x[:1]).shape == (1, 3, 256, 256)
        p0, s0 = image_metrics(y, y)
        assert torch.allclose(s0, torch.ones_like(s0), atol=1e-5) and (p0 > 100).all()
    assert retrieval(np.eye(8), np.eye(8))['top1'] == 1
    assert retrieval(np.ones((8, 8)), np.ones((8, 8)))['top5'] == 0
    cfg = dict(version=VERSION, registration_sha256=digest(regpath), targets_sha256=digest(targets),
               holdout='fov', out_size=128, width=8,
               normalizer_sha256={name: digest(Path(W)/name) for name in
                                  ['cellmeta.npz', 'led_norm.npz', 'backgrounds_p50.npy']})
    ckpt = tmp/'ckpt.pt'
    atomic_torch(dict(config=cfg, model=net.state_dict(), optimizer=opt.state_dict(),
                      epoch=0, step=2, rng=rng_state()), ckpt)
    reloaded = torch.load(ckpt, weights_only=False)
    other = UNet(8); other.load_state_dict(reloaded['model'])
    otheropt = torch.optim.AdamW(other.parameters()); otheropt.load_state_dict(reloaded['optimizer'])
    with torch.no_grad():
        assert torch.allclose(net(x), other(x), atol=2e-6)
    print(json.dumps(dict(stage='forward_backward', l1_dino=losses, patchgan=gan_losses,
                          strict_dinobloom_s=True, checkpoint_roundtrip=True)), flush=True)
    del dino, disc, net, other, net256, norm, opt, dopt, otheropt
    gc.collect()
    # End-to-end evaluator on the remaining four held-out cells, including strict B load,
    # montage, candidate ties, row mapping and idempotent completed-output replay.
    import vs_eval
    original = sys.argv
    sys.argv = ['vs_eval.py', '--checkpoint', str(ckpt), '--registration', str(regpath),
                '--out', str(tmp/'eval'), '--device', 'cpu', '--workers', '0', '--bs', '2']
    real_dino = vs_eval.Dino
    class InterruptDino(real_dino):
        calls = 0
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            self.calls += 1
            if self.calls == 2:
                os.kill(os.getpid(), signal.SIGUSR1)
            return result
    vs_eval.Dino = InterruptDino
    vs_eval.main()
    with np.load(tmp/'eval/eval_state.npz') as f:
        assert int(f['cursor']) == 2
    assert not (tmp/'eval/generated.npz').exists()
    vs_eval.Dino = real_dino
    vs_eval.main(); vs_eval.main()
    sys.argv = original
    with np.load(tmp/'eval/generated.npz') as f:
        assert f['images'].dtype == np.uint8 and f['images'].shape == (4, 128, 128, 3)
        assert np.array_equal(f['gidx'], hc[te])
    assert (tmp/'eval/montage.png').exists()
    train_signal_control(tmp, regpath)
    print(json.dumps(dict(result='PASS', real_cells=8, seconds=round(time.monotonic()-t0, 2),
                          checks=['phase_sign', 'multiprocessing_48', '128_256_geometry', 'GPUNorm_24ch', 'S_perceptual_backward',
                                  'PatchGAN_backward', 'B_eval', 'checkpoint_optimizer_rng', 'resume_sampler',
                                  'SSIM_identity', 'pessimistic_ties', 'montage', 'uint8_NPZ_row_ids', 'eval_SIGUSR1_resume', 'train_SIGUSR1_resume'])), flush=True)


def train_signal_control(tmp, regpath):
    """Exercise the real main-loop/save/signal code with a tiny synthetic transaction.

    No cell data are read and no training optimizer is stepped by this control.
    """
    import vs_train as v
    from unittest.mock import patch
    class Tiny(torch.nn.Module):
        def __init__(self, *args):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.randn(2))
    class DummyDataset:
        def __init__(self, reg, rows, size, pack):
            self.n = len(rows)
        def __len__(self):
            return self.n
    def fake_loader(ds, bs, workers, sampler=None):
        class Loader:
            def __iter__(self):
                batches = list(sampler) if sampler else [[0]]
                for _ in batches:
                    yield None
        return Loader()
    mode = dict(interrupt_train=False, interrupt_eval=False, calls=0)
    def transaction(net, *args, **kwargs):
        with torch.no_grad():
            net.weight.add_(torch.randn_like(net.weight))
        mode['calls'] += 1
        if mode['interrupt_train'] and mode['calls'] == 1:
            os.kill(os.getpid(), signal.SIGUSR1)
        return {'loss': float(net.weight.sum())}
    def evaluation(net, loader, norm, device, heartbeat):
        if mode['interrupt_eval']:
            os.kill(os.getpid(), signal.SIGUSR1)
        return None if heartbeat() else dict(l1=0., psnr=0., ssim=0.)
    def invoke(out, bs='2'):
        argv = ['vs_train.py', '--registration', str(regpath), '--out', str(out),
                '--device', 'cpu', '--wandb-mode', 'disabled', '--epochs', '2',
                '--bs', bs, '--workers', '0', '--perceptual-weight', '0', '--translate', '0']
        with patch.object(sys, 'argv', argv):
            v.main()
    with patch.multiple(v, UNet=Tiny, Pairs=DummyDataset, GPUNorm=lambda _: None,
            make_loader=fake_loader, normalize_batch=lambda *args: (torch.zeros(2, 1), torch.zeros(2, 1)),
            translate_pair=lambda x, y, r: (x, y), train_step=transaction, evaluate=evaluation):
        invoke(tmp/'control_full')
        mode.update(interrupt_train=True, calls=0)
        invoke(tmp/'control_resume')
        first = torch.load(tmp/'control_resume/ckpt.pt', weights_only=False)
        assert first['epoch'] == 0 and first['next_batch'] == 1 and first['step'] == 1
        try:
            invoke(tmp/'control_resume', bs='1')
        except ValueError as e:
            assert 'resume config mismatch' in str(e)
        else:
            raise AssertionError('changed batch size accepted')
        mode.update(interrupt_train=False, interrupt_eval=True)
        invoke(tmp/'control_resume')
        mid = torch.load(tmp/'control_resume/ckpt.pt', weights_only=False)
        assert mid['epoch'] == 1 and mid['eval_pending']
        mode.update(interrupt_eval=False)
        invoke(tmp/'control_resume')
        full = torch.load(tmp/'control_full/ckpt.pt', weights_only=False)
        resumed = torch.load(tmp/'control_resume/ckpt.pt', weights_only=False)
        assert full['step'] == resumed['step'] == 4
        assert full['epoch'] == resumed['epoch'] == 2
        assert torch.equal(full['model']['weight'], resumed['model']['weight'])
        assert torch.equal(full['rng']['torch'], resumed['rng']['torch'])
    print(json.dumps(dict(stage='signal_control', training_cursor=True, pending_eval=True,
          changed_config_rejected=True, resumed_matches_uninterrupted=True)), flush=True)


if __name__ == '__main__':
    main()
