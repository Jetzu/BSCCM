"""In-domain LF encoder pretraining with SIGReg or DINO self-distillation."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import copy
import json
import math
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from bsccm.data import W
from bsccm.gpu_norm import RawSet, GPUNorm
from bsccm.ssl import make_views_batch, ssl_loss


class DINOHead(nn.Module):
    def __init__(self, features, prototypes):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(features, 2048), nn.GELU(),
                                 nn.Linear(2048, 2048), nn.GELU(), nn.Linear(2048, 256))
        self.last = nn.utils.parametrizations.weight_norm(nn.Linear(256, prototypes, bias=False))
        nn.init.ones_(self.last.parametrizations.weight.original0)

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x).float(), dim=-1))


def local_crop(view, gen):
    """Crop a 64px square per cell, jittering its centre by up to eight pixels."""
    b, _, h, w = view.shape
    if min(h, w) < 64:
        raise ValueError('local crops require global views of at least 64x64')
    starts = []
    for size in (h, w):
        mid = (size - 64) // 2
        starts.append(torch.randint(max(0, mid - 8), min(size - 64, mid + 8) + 1,
                                    (b,), generator=gen, device=view.device))
    pixels = torch.arange(64, device=view.device)
    y, x = (start[:, None] + pixels for start in starts)
    crop = view.permute(0, 2, 3, 1)[torch.arange(b, device=view.device)[:, None, None],
                                     y[:, :, None], x[:, None, :]]
    return crop.permute(0, 3, 1, 2)


def cosine_value(first, last, step, total_steps):
    progress = step / max(1, total_steps - 1)
    return first + (last - first) * (1 - math.cos(math.pi * progress)) / 2


def teacher_temperature(epoch, epochs):
    warm_epochs = math.ceil(0.3 * epochs)
    if epoch >= warm_epochs:
        return 0.07
    return 0.04 + 0.03 * epoch / max(1, warm_epochs - 1)


def dino_loss(student, teacher, center, temperature):
    """FP32 cross-entropy over ordered cross-view pairs, equally weighted."""
    with torch.no_grad():
        logits = torch.stack(teacher).float()
        log_q = ((logits - center) / temperature).log_softmax(-1)
        q = log_q.exp()
        entropy = -(q * log_q).sum(-1).mean()
        batch_center = logits.mean(dim=(0, 1))
        marginal = q.mean(dim=(0, 1))
    log_p = [(s.float() / 0.1).log_softmax(-1) for s in student]
    pairs = [-(target * pred).sum(-1).mean() for i, target in enumerate(q)
             for j, pred in enumerate(log_p) if i != j]
    return torch.stack(pairs).mean(), entropy, batch_center, marginal


@torch.no_grad()
def update_teacher(student, teacher, momentum):
    # BN buffers accumulate the teacher's own two global forwards in train mode.
    for src, dst in zip(student.parameters(), teacher.parameters()):
        dst.lerp_(src, 1 - momentum)


class SSLCells(RawSet):
    """Raw int16 records only (normalisation + views run batched on the GPU). Index = (row-offset, epoch) from EpochBatches."""
    def __getitem__(self, index):
        i, _epoch = index
        return super().__getitem__(i)


class EpochBatches:
    """Deterministic shuffle, skip already-committed batches without reading them."""
    def __init__(self, n, bs, seed):
        self.n, self.bs, self.seed = n, bs, seed
        self.epoch, self.start = 0, 0

    @property
    def total(self):
        return self.n // self.bs

    def __len__(self):
        return self.total - self.start

    def __iter__(self):
        order = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for b in range(self.start, self.total):
            yield [(i, self.epoch) for i in order[b * self.bs:(b + 1) * self.bs]]


def atomic_save(obj, path):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    torch.cuda.set_rng_state_all(s['cuda'])


def loader_options(workers, seed):
    kw = dict(num_workers=workers, pin_memory=True,
              generator=torch.Generator().manual_seed(seed))
    if workers:
        kw.update(persistent_workers=True, prefetch_factor=6)
    return kw


@torch.no_grad()
def health_metrics(model, loader, gnorm):
    model.eval()
    features = []
    for raw, ph, g in loader:
        x = gnorm(raw.cuda(non_blocking=True), ph.cuda(non_blocking=True), g.cuda(non_blocking=True), 'all', True)
        x = torch.cat((x, torch.ones_like(x)), 1)                      # full view: all 24 channels present
        with torch.autocast('cuda', dtype=torch.bfloat16):
            h = model(x.contiguous(memory_format=torch.channels_last))
        features.append(h.float())
    h = torch.cat(features)
    h = h - h.mean(0)
    # Entropy of singular values, as requested (not their squares).
    sv = torch.linalg.svdvals(h)
    total = sv.sum()
    p = sv / total.clamp_min(1e-12)
    rank = torch.exp(-(p * p.clamp_min(1e-12).log()).sum())
    rank = torch.where(total > 1e-12, rank, rank.new_zeros(()))
    unit = h / h.square().sum(0).sqrt().clamp_min(1e-12)
    corr = unit.T @ unit
    d = corr.shape[0]
    if d < 2:
        raise ValueError('encoder health requires at least two features')
    offdiag = (corr.abs().sum() - corr.diag().abs().sum()) / (d * (d - 1))
    model.train()
    return {'effective_rank': rank.item(), 'mean_abs_offdiag_corr': offdiag.item()}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', choices=['aug', 'xillum'], required=True)
    p.add_argument('--objective', choices=['sigreg', 'dino'], default='sigreg')
    p.add_argument('--prototypes', type=int, default=4096, help='DINO prototype count')
    p.add_argument('--local-crops', type=int, default=0, help='additional 64px student views for DINO')
    p.add_argument('--out', required=True)
    p.add_argument('--arch', default='resnet50')
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--bs', type=int, default=512)
    p.add_argument('--views', type=int, default=2)
    p.add_argument('--lr', type=float, default=2e-3)
    p.add_argument('--lam', type=float, default=0.05)
    p.add_argument('--proj-dim', type=int, default=256)
    p.add_argument('--workers', type=int, default=15)
    p.add_argument('--rows', choices=['all', 'train'], default='all')
    p.add_argument('--exclude-batch', type=int, nargs='+', default=[])
    p.add_argument('--max-steps', type=int, default=0, help='global step limit; leaves resumable ckpt, no final encoder')
    p.add_argument('--resume', default='auto', help='auto (out/ckpt.pt), none, or checkpoint path')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if (a.epochs < 1 or a.bs < 2 or a.views < 2 or a.workers < 0 or a.lr <= 0
            or a.proj_dim < 1 or not 0 <= a.lam <= 1 or a.max_steps < 0 or a.seed < 0):
        p.error('invalid training parameter')
    if a.arm == 'xillum' and a.views > 6:
        p.error('xillum supports 2..6 views (>=4 channels each)')
    if a.objective == 'dino' and (a.views != 2 or a.prototypes < 2 or a.local_crops < 0):
        p.error('dino requires two global views, >=2 prototypes and nonnegative --local-crops')
    if a.objective == 'sigreg' and (a.local_crops or a.prototypes != 4096):
        p.error('--local-crops and --prototypes are DINO options')
    return a


def main():
    a = parse_args()
    dino = a.objective == 'dino'
    stop = False

    def on_term(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, on_term)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('training requires a CUDA device with bf16 support')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = out / 'ckpt.pt'
    resume_path = ckpt_path if a.resume == 'auto' else (None if a.resume == 'none' else Path(a.resume))
    if a.resume == 'none' and (ckpt_path.exists() or (out / 'encoder_final.pt').exists()):
        raise FileExistsError('refusing to overwrite an existing run with --resume none')
    if a.resume not in ('auto', 'none') and not resume_path.is_file():
        raise FileNotFoundError(resume_path)
    if a.resume == 'auto' and not ckpt_path.exists() and (out / 'encoder_final.pt').exists():
        raise FileNotFoundError('final encoder exists without ckpt.pt; use a fresh output directory')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    with np.load(W + '/targets_v3.npz') as t:
        unknown = set(a.exclude_batch) - set(np.unique(t['batch']).tolist())
        if unknown:
            raise ValueError(f'unknown excluded batches: {sorted(unknown)}')
        keep = np.ones(len(t['split']), dtype=bool) if a.rows == 'all' else t['split'] == 0
        keep &= ~np.isin(t['batch'], a.exclude_batch)
    rows = np.flatnonzero(keep)
    if len(rows) < max(1024, a.bs):
        raise ValueError('need >=1024 eligible cells for fixed health batch and >=bs for training')
    sampler = EpochBatches(len(rows), a.bs, a.seed)
    train = torch.utils.data.DataLoader(SSLCells(rows),
                                       batch_sampler=sampler, **loader_options(a.workers, a.seed))
    health_rows = np.random.default_rng(a.seed).choice(rows, 1024, replace=False)
    health = torch.utils.data.DataLoader(RawSet(health_rows), batch_size=min(a.bs, 64),
                                        **loader_options(a.workers, a.seed + 1))
    gnorm = GPUNorm('cuda')
    model = timm.create_model(a.arch, pretrained=False, in_chans=48, num_classes=0)
    if dino:
        projector = DINOHead(model.num_features, a.prototypes)
    else:
        projector = nn.Sequential(nn.Linear(model.num_features, 2048), nn.BatchNorm1d(2048), nn.GELU(),
                                  nn.Linear(2048, 2048), nn.BatchNorm1d(2048), nn.GELU(),
                                  nn.Linear(2048, a.proj_dim))
    model = model.cuda().to(memory_format=torch.channels_last)
    projector = projector.cuda()
    if dino:
        teacher = copy.deepcopy(model).requires_grad_(False)
        teacher_head = copy.deepcopy(projector).requires_grad_(False)
        center = torch.zeros(a.prototypes, device=next(teacher.parameters()).device)
        probability_sum = torch.zeros_like(center)
        momentum = 0.996
    params = list(model.parameters()) + list(projector.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.04 if dino else 0.05)
    total_steps, warm_steps = a.epochs * sampler.total, 2 * sampler.total

    def lr_factor(step):
        if step < warm_steps:
            return (step + 1) / warm_steps
        progress = min(1.0, (step - warm_steps) / max(1, total_steps - warm_steps))
        return 0.5 * (1 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    epoch = batch = step = 0
    # Three metric sums weighted by batch size, image count, elapsed epoch seconds.
    accum = np.zeros(5, dtype=np.float64)
    resume_rng = None
    options = vars(a).copy()
    if not dino:  # Preserve the pre-DINO config for legacy checkpoints and W&B.
        for key in ('objective', 'local_crops', 'prototypes'):
            del options[key]
    config = {k: v for k, v in options.items() if k not in ('out', 'resume', 'max_steps', 'workers')}
    if resume_path is not None and resume_path.exists():
        s = torch.load(resume_path, map_location='cpu', weights_only=False)
        if s['config'] != config or not np.array_equal(s['rows'], rows):
            raise ValueError('resume configuration or eligible rows differ from checkpoint')
        model.load_state_dict(s['model'])
        projector.load_state_dict(s['projector'])
        if dino:
            teacher.load_state_dict(s['teacher'])
            teacher_head.load_state_dict(s['teacher_head'])
            center.copy_(s['center'])
            probability_sum.copy_(s['probability_sum'])
            momentum = s['momentum']
        opt.load_state_dict(s['opt'])
        sched.load_state_dict(s['sched'])
        epoch, batch, step = s['epoch'], s['batch'], s['step']
        accum = np.asarray(s['accum'], dtype=np.float64)
        resume_rng = s['rng']
        print(f'resumed epoch={epoch} next_batch={batch} step={step}', flush=True)
        del s
    import wandb
    run = wandb.init(project=os.environ.get('WANDB_PROJECT', 'BSCCM'),
                     name=out.resolve().name, id=out.resolve().name, mode='offline',
                     resume='allow', dir=str(out),
                     config={**options, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
                             'n_rows': len(rows), 'transductive': a.rows == 'all'})
    if resume_rng is not None:
        restore_rng(resume_rng)
        del resume_rng
    last_save = time.monotonic()

    def save():
        nonlocal last_save
        extra = dict(teacher=teacher.state_dict(), teacher_head=teacher_head.state_dict(),
                     center=center, probability_sum=probability_sum, momentum=momentum) if dino else {}
        atomic_save(dict(model=model.state_dict(), projector=projector.state_dict(),
                         opt=opt.state_dict(), sched=sched.state_dict(), epoch=epoch,
                         batch=batch, step=step, rng=rng_state(), accum=accum,
                         config=config, rows=rows, **extra), ckpt_path)
        last_save = time.monotonic()

    model.train()
    projector.train()
    while epoch < a.epochs:
        if dino:
            # Freeze only the prototype magnitudes. Directions still get gradients.
            projector.last.parametrizations.weight.original0.requires_grad_(epoch > 0)
            temperature = teacher_temperature(epoch, a.epochs)
        if stop or (a.max_steps and step >= a.max_steps):
            save()
            run.finish()
            return
        sampler.epoch, sampler.start = epoch, batch
        tick = time.monotonic()
        for raw, ph, g in train:
            x = gnorm(raw.cuda(non_blocking=True), ph.cuda(non_blocking=True), g.cuda(non_blocking=True), 'all', True)
            gen = torch.Generator(device='cuda').manual_seed(a.seed * 1_000_003 + step)   # deterministic per step (resume-safe)
            views = make_views_batch(x, a.arm, a.views, gen)
            opt.zero_grad(set_to_none=True)
            if dino:
                for group in opt.param_groups:
                    group['weight_decay'] = cosine_value(0.04, 0.4, step, total_steps)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    targets = [teacher_head(teacher(v.contiguous(memory_format=torch.channels_last)))
                               for v in views]
                for i in range(a.local_crops):
                    views.append(local_crop(views[i % 2], gen))
            zs = []
            # Separate forwards preserve per-view BN statistics, with one joint loss.
            for x in views:
                x = x.cuda(non_blocking=True).contiguous(memory_format=torch.channels_last)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    zs.append(projector(model(x)))
            if dino:
                loss, entropy, batch_center, marginal = dino_loss(zs, targets, center, temperature)
                components = (entropy, math.log(a.prototypes) - entropy)
            else:
                loss, pred, reg = ssl_loss(zs, a.lam)
                components = (pred.detach(), reg.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite loss at step {step}')
            loss.backward()
            nn.utils.clip_grad_norm_(params, 3.0, error_if_nonfinite=True)
            opt.step()
            if dino:
                momentum = cosine_value(0.996, 1., step, total_steps)
                update_teacher(model, teacher, momentum)
                update_teacher(projector, teacher_head, momentum)
                center.lerp_(batch_center, 0.1)
                probability_sum.add_(marginal, alpha=a.bs)
            sched.step()
            vals = torch.stack((loss.detach(), *components)).cpu().numpy()
            now = time.monotonic()
            accum[:3] += vals * a.bs
            accum[3] += a.bs
            accum[4] += now - tick
            step += 1
            batch += 1
            if stop or (a.max_steps and step >= a.max_steps):
                save()
                run.finish()
                return
            if now - last_save >= 20 * 60:
                save()
            tick = time.monotonic()
        metrics = health_metrics(teacher if dino else model, health, gnorm)
        if accum[3] == 0:
            raise RuntimeError('epoch has no accumulated training samples')
        metric1, metric2 = ('teacher_entropy', 'teacher_kl_uniform') if dino else ('pred', 'sigreg')
        record = dict(epoch=epoch + 1, loss=float(accum[0] / accum[3]),
                      **{metric1: float(accum[1] / accum[3]), metric2: float(accum[2] / accum[3])},
                      secs=float(accum[4]), imgs_per_s=float(accum[3] / max(accum[4], 1e-12)),
                      **metrics)
        if dino:
            # Pooled prototype usage distinguishes sharp, diverse targets from collapse.
            usage = probability_sum / accum[3]
            record['teacher_marginal_kl_uniform'] = (torch.special.xlogy(usage, usage).sum()
                                                     + math.log(a.prototypes)).item()
            probability_sum.zero_()
        print(json.dumps(record, allow_nan=False), flush=True)
        run.log(record, step=step)
        epoch += 1
        batch = 0
        accum = np.zeros(5, dtype=np.float64)
        save()
        if stop:
            run.finish()
            return
    # Export encoder only, CPU tensors; retain just the latest full checkpoint.
    encoder = teacher if dino else model
    atomic_save({k: v.detach().cpu() for k, v in encoder.state_dict().items()}, out / 'encoder_final.pt')
    run.finish()


if __name__ == '__main__':
    main()
