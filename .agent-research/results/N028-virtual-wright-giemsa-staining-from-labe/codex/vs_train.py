"""Single-GPU, resumable paired LF -> virtual Wright/Giemsa training."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import hashlib
import json
import os
import random
import signal
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from vs_common import (W, VERSION, UNet, PatchGAN, Dino, Pairs, GPUNorm, EpochBatches,
    amp, atomic_torch, digest, image_metrics, make_loader, normalize_batch, perceptual_loss,
    read_registration, restore_rng, rng_state, robust_l1, translate_pair)


def train_step(net, disc, dino, opt, dopt, x, y, perceptual_weight=.1, gan_weight=.01):
    """One complete optimizer transaction; signals are serviced after it returns."""
    if disc is not None:
        disc.requires_grad_(True)
    with amp(x.device):
        pred = net(x)
        dloss = pred.new_zeros(())
        if disc is not None:
            dloss = .5 * (F.relu(1-disc(x, y)).mean() + F.relu(1+disc(x, pred.detach())).mean())
    if disc is not None:
        dopt.zero_grad(set_to_none=True)
        dloss.backward()
        nn.utils.clip_grad_norm_(disc.parameters(), 5., error_if_nonfinite=True)
        dopt.step()
        disc.requires_grad_(False)
    with amp(x.device):
        l1 = robust_l1(pred, y)
        feature = perceptual_loss(dino, pred, y) if perceptual_weight else l1.new_zeros(())
        adv = -disc(x, pred).mean() if disc is not None else l1.new_zeros(())
        loss = l1 + perceptual_weight*feature + gan_weight*adv
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 5., error_if_nonfinite=True)
    opt.step()
    return dict(loss=float(loss.detach()), l1=float(l1.detach()), perceptual=float(feature.detach()),
                gan=float(adv.detach()), discriminator=float(dloss.detach()))


@torch.no_grad()
def evaluate(net, loader, norm, device, heartbeat):
    net.eval()
    sums = np.zeros(3, np.float64)
    n = 0
    for batch in loader:
        if heartbeat():
            return None
        x, y = normalize_batch(batch, norm, device)
        with amp(device):
            pred = net(x)
        psnr, ssim = image_metrics(pred, y)
        sums += [float((pred.float()-y).abs().sum()) / (3*y.shape[-1]**2),
                 float(psnr.sum()), float(ssim.sum())]
        n += len(x)
    net.train()
    return dict(zip(['l1', 'psnr', 'ssim'], (sums / n).tolist()))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--registration', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--pack', default=W+'/pack')
    p.add_argument('--targets', default=W+'/targets_v6.npz')
    p.add_argument('--holdout', choices=['fov', 'batch0', 'batch1'], default='fov')
    p.add_argument('--out-size', type=int, choices=[128, 256], default=128)
    p.add_argument('--width', type=int, default=32)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--bs', type=int, default=32)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--perceptual', choices=['S', 'B'], default='S')
    p.add_argument('--dino-path', default='')
    p.add_argument('--perceptual-weight', type=float, default=.1)
    p.add_argument('--patchgan', action='store_true')
    p.add_argument('--gan-weight', type=float, default=.01)
    p.add_argument('--translate', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-minutes', type=float, default=15)
    p.add_argument('--eval-every', type=int, default=1, help='epochs; no checkpoint selection on test')
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--wandb-mode', choices=['offline', 'online', 'disabled'], default='offline')
    return p


def main():
    p = parser(); a = p.parse_args()
    if (a.width < 8 or a.width % 8 or a.bs < 1 or a.epochs < 1 or a.workers < 0 or a.lr <= 0
        or a.perceptual_weight < 0 or a.gan_weight < 0 or not 0 <= a.translate <= 8
        or not 0 < a.save_minutes <= 20 or a.eval_every < 1):
        p.error('invalid training parameter')
    stop = False
    def on_signal(signum, frame):
        nonlocal stop
        stop = True
    for s in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(s, on_signal)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    if a.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no silent CPU training fallback')
    torch.backends.cudnn.benchmark = True
    reg, tr, te = read_registration(a.registration, a.targets, a.pack, a.holdout)
    dino_path = a.dino_path or f'{W}/models/DinoBloom-{a.perceptual}.pth'
    cfg = {k: v for k, v in vars(a).items() if k not in
           ('out', 'workers', 'save_minutes', 'eval_every', 'wandb_mode', 'registration', 'targets', 'dino_path')}
    cfg.update(version=VERSION, registration_sha256=digest(a.registration), targets_sha256=digest(a.targets),
               dino_sha256=digest(dino_path), normalizer_sha256={name: digest(Path(W)/name)
                   for name in ['cellmeta.npz', 'led_norm.npz', 'backgrounds_p50.npy']})
    ckpt_path = out/'ckpt.pt'
    saved = torch.load(ckpt_path, map_location='cpu', weights_only=False) if ckpt_path.exists() else None
    if saved is not None and saved['config'] != cfg:
        changed = [k for k in cfg.keys() | saved['config'].keys() if cfg.get(k) != saved['config'].get(k)]
        raise ValueError(f'resume config mismatch: {changed}; use a fresh --out')
    net = UNet(a.width, a.out_size).to(a.device, memory_format=torch.channels_last)
    disc = PatchGAN().to(a.device, memory_format=torch.channels_last) if a.patchgan else None
    dino = Dino(a.perceptual, dino_path).to(a.device) if a.perceptual_weight else None
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=.01)
    dopt = torch.optim.AdamW(disc.parameters(), lr=a.lr, betas=(.5, .999)) if disc is not None else None
    epoch = next_batch = step = 0
    eval_pending = False
    segment = {}
    if saved is not None:
        net.load_state_dict(saved['model'], strict=True); opt.load_state_dict(saved['optimizer'])
        if disc is not None:
            disc.load_state_dict(saved['discriminator']); dopt.load_state_dict(saved['d_optimizer'])
        epoch, next_batch, step = saved['epoch'], saved['next_batch'], saved['step']
        eval_pending, segment = saved['eval_pending'], saved['segment']
    sampler = EpochBatches(len(tr), a.bs, a.seed)
    train_loader = make_loader(Pairs(reg, tr, a.out_size, a.pack), a.bs, a.workers, sampler)
    test_loader = make_loader(Pairs(reg, te, a.out_size, a.pack), a.bs, a.workers)
    norm = GPUNorm(a.device)
    # Logging initialization and model construction may consume RNG; restore last.
    run_id = saved['wandb_id'] if saved is not None else hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:12]
    wb = None
    if a.wandb_mode != 'disabled':
        import wandb
        wb = wandb.init(project=os.environ.get('WANDB_PROJECT', 'BSCCM'), id=run_id,
                        name=f'N028-{a.holdout}-{a.seed}', dir=str(out), mode=a.wandb_mode,
                        config=cfg, resume='allow' if a.wandb_mode == 'online' else None)
    with open(out/'sync_lineage.jsonl', 'a') as f:
        f.write(json.dumps(dict(wandb_id=run_id, job_id=os.environ.get('SLURM_JOB_ID'),
             wandb_dir=str(out.resolve()), resumed=saved is not None, time=time.time()))+'\n')
    if saved is not None:
        restore_rng(saved['rng'])
    last_save = time.monotonic()
    def save():
        nonlocal last_save
        atomic_torch(dict(config=cfg, model=net.state_dict(), optimizer=opt.state_dict(),
            discriminator=disc.state_dict() if disc is not None else None,
            d_optimizer=dopt.state_dict() if dopt is not None else None,
            epoch=epoch, next_batch=next_batch, step=step, rng=rng_state(), wandb_id=run_id,
            eval_pending=eval_pending, segment=segment), ckpt_path)
        last_save = time.monotonic()
    def heartbeat():
        if stop or time.monotonic()-last_save >= a.save_minutes*60:
            save()
        return stop
    print(json.dumps(dict(event='start', train_pairs=len(tr), test_pairs=len(te), step=step,
                          resumed=saved is not None, device=a.device)), flush=True)
    save()  # Even an immediate signal or first-batch preemption has a valid checkpoint.
    while (epoch < a.epochs or eval_pending) and not stop:
        if eval_pending:
            ev = evaluate(net, test_loader, norm, a.device, heartbeat)
            if ev is None:
                break
            log = dict(event='eval', epoch=epoch, step=step, heldout=ev, **segment)
            with open(out/'metrics.jsonl', 'a') as f:
                f.write(json.dumps(log)+'\n'); f.flush()
            print(json.dumps(log), flush=True)
            if wb is not None:
                wb.log({**{f'heldout/{k}': v for k, v in ev.items()}, 'epoch': epoch}, step=step)
            eval_pending = False
            save()
            continue
        sampler.epoch, sampler.start = epoch, next_batch
        net.train()
        acc = {}; count = 0; t0 = time.monotonic()
        for batch in train_loader:
            if heartbeat():
                break
            x, y = normalize_batch(batch, norm, a.device)
            x, y = translate_pair(x, y, a.translate)
            values = train_step(net, disc, dino, opt, dopt, x, y, a.perceptual_weight, a.gan_weight)
            step += 1; next_batch += 1
            count += len(x)
            for k, v in values.items():
                acc[k] = acc.get(k, 0.) + v*len(x)
            if heartbeat():
                break
        if stop:
            break
        if next_batch != sampler.total:
            raise RuntimeError('epoch cursor does not match completed batches')
        epoch += 1; next_batch = 0
        segment = dict(train={k: v/count for k, v in acc.items()} if count else {},
                       train_pairs_this_segment=count, seconds=time.monotonic()-t0)
        eval_pending = epoch % a.eval_every == 0 or epoch == a.epochs
        save()
    save()
    if wb is not None:
        wb.finish()
    print(json.dumps(dict(event='stopped' if stop else 'complete', epoch=epoch, step=step,
                          checkpoint=str(ckpt_path))), flush=True)


if __name__ == '__main__':
    main()
