"""Masked image reconstruction using only the source draw of an N087 fold."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

# Support both the staged writer directory and installation under repo/scripts.
HERE = Path(__file__).resolve()
REPO = next(p for p in HERE.parents if (p / 'bsccm/gpu_norm.py').is_file())
sys.path.insert(0, str(REPO))
from bsccm.data import W
from bsccm.gpu_norm import GPUNorm, RawSet


class Encoder(nn.Sequential):
    """128 -> 64 -> 32 -> 16 -> 8, followed by a linear spatial bottleneck."""
    def __init__(self, dim):
        layers = []
        channels = (24, 32, 48, 64, 96)
        for cin, cout in zip(channels, channels[1:]):
            layers.extend([nn.Conv2d(cin, cout, 4, stride=2, padding=1),
                           nn.GroupNorm(8, cout), nn.GELU()])
        super().__init__(*layers, nn.Flatten(), nn.Linear(96 * 8 * 8, dim))


class ImageAE(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.encoder = Encoder(dim)
        layers = [nn.Linear(dim, 96 * 8 * 8), nn.Unflatten(1, (96, 8, 8))]
        channels = (96, 64, 48, 32, 24)
        for i, (cin, cout) in enumerate(zip(channels, channels[1:])):
            layers.append(nn.ConvTranspose2d(cin, cout, 4, stride=2, padding=1))
            if i < len(channels) - 2:
                layers.extend([nn.GroupNorm(8, cout), nn.GELU()])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))


def source_partition(path, protocol, seed):
    excluded = int(protocol[-1])
    with np.load(path) as t:
        batch = t['batch']
        if set(np.unique(batch)) != {0, 1}:
            raise ValueError('expected draw IDs 0 and 1 in targets_v3')
        eligible = np.flatnonzero(batch != excluded)  # SSL --rows all --exclude-batch
        groups = np.column_stack((t['slide'][eligible], t['fov'][eligible]))
    unique, index = np.unique(groups, axis=0, return_inverse=True)
    if len(unique) < 2:
        raise ValueError('need at least two source fields of view')
    chosen = np.random.default_rng(seed).choice(len(unique), max(1, round(.01 * len(unique))),
                                                replace=False)
    validation = np.isin(index.reshape(-1), chosen)
    train, val = eligible[~validation], eligible[validation]
    assert np.all(batch[train] == 1 - excluded) and np.all(batch[val] == 1 - excluded)
    return train, val


def normalise(norm, sample, device):
    raw, phase, rows = sample
    return norm(raw.to(device, non_blocking=True), phase.to(device, non_blocking=True),
                rows.to(device, non_blocking=True), 'all', True, shift=0)


def masked_input(x, fraction, seed):
    """Drop exactly round(fraction * 64) spatial patches per cell across all channels."""
    valid = torch.isfinite(x)
    clean = torch.where(valid, x, 0.)
    if fraction == 0:
        return clean, clean, valid
    n = len(x)
    gen = torch.Generator(device=x.device).manual_seed(seed)
    count = max(1, min(63, round(fraction * 64)))
    indices = torch.rand(n, 64, generator=gen, device=x.device).argsort(dim=1)[:, :count]
    mask = torch.zeros(n, 64, dtype=torch.bool, device=x.device).scatter_(1, indices, True)
    mask = mask.view(n, 1, 8, 8).repeat_interleave(16, 2).repeat_interleave(16, 3)
    return clean.masked_fill(mask, 0.), clean, valid & mask


class ImageBatches:
    """Keep the last short batch; resume without reopening completed image records."""
    def __init__(self, n, bs, shuffle):
        self.n, self.bs, self.shuffle = n, bs, shuffle
        self.seed = self.start = 0

    def __iter__(self):
        order = (np.random.default_rng(self.seed).permutation(self.n) if self.shuffle
                 else np.arange(self.n))
        for pos in range(self.start * self.bs, self.n, self.bs):
            yield order[pos:pos + self.bs].tolist()

    def __len__(self):
        return math.ceil(self.n / self.bs) - self.start


def atomic_write(path, write):
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', choices=('b1to0', 'b0to1'), required=True)
    p.add_argument('--dim', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, help='default approximately 5.4M source cell presentations')
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--mask-fraction', type=float, default=.25)
    p.add_argument('--l1', type=float, default=0., help='weight on masked reconstruction MAE')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-steps', type=int, default=0, help='stop after this many optimizer steps with resumable ckpt')
    p.add_argument('--no-wandb', action='store_true', help='local smoke tests; production logs offline by default')
    a = p.parse_args(argv)
    if (min(a.dim, a.bs) < 1 or (a.epochs is not None and a.epochs < 1) or a.workers < 0
            or a.max_steps < 0 or a.seed < 0 or not 0 <= a.mask_fraction < 1
            or not 0 < a.lr < math.inf or not 0 <= a.l1 < math.inf):
        p.error('invalid training options')
    return a


def train(a):
    stop = False

    def on_signal(signum, frame):
        nonlocal stop
        stop = True

    handlers = {s: signal.signal(s, on_signal) for s in (signal.SIGTERM, signal.SIGUSR1)}
    run = None
    try:
        a.out.mkdir(parents=True, exist_ok=True)
        ckpt = a.out / 'ckpt.pt'
        torch.manual_seed(a.seed)
        train_rows, val_rows = source_partition(Path(W) / 'targets_v3.npz', a.protocol, a.seed)
        epochs = a.epochs or max(1, round(5_400_000 / len(train_rows)))
        config = {k: v for k, v in vars(a).items()
                  if k not in ('out', 'device', 'no_wandb', 'max_steps', 'workers')}
        config.update(epochs=epochs, format_version=1,
                      train_sha256=hashlib.sha256(train_rows.tobytes()).hexdigest(),
                      val_sha256=hashlib.sha256(val_rows.tobytes()).hexdigest())
        norm = GPUNorm(a.device)
        loaders, samplers = {}, {}
        for phase, rows in [('train', train_rows), ('val', val_rows)]:
            sampler = ImageBatches(len(rows), a.bs, shuffle=phase == 'train')
            samplers[phase] = sampler
            loaders[phase] = DataLoader(RawSet(rows), batch_sampler=sampler, num_workers=a.workers,
                                        persistent_workers=a.workers > 0, pin_memory=a.device == 'cuda')
        model = ImageAE(a.dim).to(a.device, memory_format=torch.channels_last)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr)
        state = dict(epoch=0, phase='train', batch=0, steps=0, squared_error=0.,
                     valid_pixels=0, log=[], train_mse=None)
        if ckpt.exists():
            saved = torch.load(ckpt, map_location='cpu', weights_only=True)
            if saved['config'] != config:
                raise ValueError('checkpoint config or source rows disagree with this invocation')
            model.load_state_dict(saved['model'])
            opt.load_state_dict(saved['opt'])
            state = saved['state']
            del saved
        elif (a.out / 'encoder_final.pt').exists():
            raise FileExistsError('found final encoder without its training checkpoint')

        if not a.no_wandb and state['epoch'] < epochs:
            import wandb
            run = wandb.init(mode='offline', project=os.environ.get('WANDB_PROJECT', 'BSCCM'),
                             name=a.out.resolve().name, id=a.out.resolve().name,
                             dir=os.environ.get('WANDB_DIR', str(a.out)),
                             config={**config, 'slurm_job_id': os.environ.get('SLURM_JOB_ID')})
        last_save = time.monotonic()

        def save():
            nonlocal last_save
            payload = dict(config=config, state=state, model=model.state_dict(), opt=opt.state_dict())
            atomic_write(ckpt, lambda f: torch.save(payload, f))
            last_save = time.monotonic()

        def write_log():
            data = json.dumps(state['log'], allow_nan=False, indent=2).encode() + b'\n'
            atomic_write(a.out / 'log.json', lambda f: f.write(data))

        write_log()  # ckpt is authoritative if a signal interrupted the last log write
        while state['epoch'] < epochs:
            if stop or (a.max_steps and state['steps'] >= a.max_steps):
                save()
                return
            phase = state['phase']
            training = phase == 'train'
            model.train(training)
            sampler = samplers[phase]
            sampler.start, sampler.seed = state['batch'], a.seed + state['epoch']
            for sample in loaders[phase]:
                # No worker randomness. A batch's mask is a function only of its committed index.
                mask_seed = ((a.seed + 1) * 1_000_003 + state['steps'] if training
                             else a.seed + state['batch'])
                x, target, mask = masked_input(normalise(norm, sample, a.device), a.mask_fraction, mask_seed)
                with torch.set_grad_enabled(training):
                    with torch.autocast(a.device, dtype=torch.bfloat16):
                        pred = model(x.contiguous(memory_format=torch.channels_last))
                    error = (pred.float() - target).masked_fill(~mask, 0.)
                    count = mask.sum()
                    sse = error.square().sum()
                    loss = sse / count
                    if a.l1 and training:
                        loss = loss + a.l1 * error.abs().sum() / count
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f'nonfinite loss in {phase} batch {state["batch"]}')
                    if training:
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                        state['steps'] += 1
                state['batch'] += 1
                state['squared_error'] += sse.detach().item()
                state['valid_pixels'] += count.item()
                if stop or (a.max_steps and state['steps'] >= a.max_steps):
                    save()
                    return
                if time.monotonic() - last_save >= 20 * 60:
                    save()
            mse = state['squared_error'] / state['valid_pixels']
            state.update(batch=0, squared_error=0., valid_pixels=0)
            if training:
                state.update(train_mse=mse, phase='val')
            else:
                state['epoch'] += 1
                record = dict(epoch=state['epoch'], train_mse=state['train_mse'], val_mse=mse,
                              steps=state['steps'], train_cells=len(train_rows), val_cells=len(val_rows))
                state['log'].append(record)
                state.update(phase='train', train_mse=None)
                save()
                write_log()
                print(json.dumps(record), flush=True)
                if run:
                    run.log(record, step=state['steps'])
        encoder = dict(dim=a.dim, protocol=a.protocol, model=model.encoder.cpu().state_dict())
        atomic_write(a.out / 'encoder_final.pt', lambda f: torch.save(encoder, f))
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if run is not None:
            run.finish()


if __name__ == '__main__':
    train(parse_args())
