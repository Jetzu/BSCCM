"""Shared N028 geometry, data access, models and checkpoint helpers."""
import sys
from pathlib import Path

# Works here and after copying the Python files to repo/scripts/.
REPO = next((p for p in Path(__file__).resolve().parents if (p / 'bsccm/gpu_norm.py').is_file()), None)
if REPO is None:
    raise RuntimeError('Place scripts inside the BSCCM repository')
sys.path.insert(0, str(REPO))

import contextlib
import hashlib
import json
import os
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import affine_transform, gaussian_filter
from bsccm.data import W
from bsccm.gpu_norm import GPUNorm, RawSet

VERSION = 1


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def atomic_torch(obj, path):
    path = Path(path)
    with open(str(path) + '.tmp', 'wb') as f:
        torch.save(obj, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(path) + '.tmp', path)


def atomic_npz(path, **arrays):
    with open(str(path) + '.tmp', 'wb') as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(path) + '.tmp', path)


def split_rows(targets, gidx, holdout):
    with np.load(targets) as t:
        batch, split = t['batch'][gidx], t['split'][gidx]
        if holdout == 'fov':
            return np.flatnonzero(split == 0), np.flatnonzero(split == 2)
        held = int(holdout[-1])
        return np.flatnonzero(batch == 1 - held), np.flatnonzero(batch == held)


def registered_rgb(image, scale, dy=0, dx=0, size=128):
    """Shift applies TO the 128px GS image, with no wrapping.

    source = centre + (output displacement - shift) * (398*scale/128).
    Identical pixel-centre geometry for registration, training and evaluation.
    """
    image = np.asarray(image, np.float32)
    h, w = image.shape[:2]
    step = h * float(scale) / size
    sigma = max(0., (step - 1) / 2)
    if sigma:
        image = gaussian_filter(image, (sigma, sigma, 0) if image.ndim == 3 else sigma)
    offset = np.array([(h - 1) / 2, (w - 1) / 2]) + (0.5 - size / 2) * step
    offset -= np.array([dy, dx]) * h * float(scale) / 128
    if image.ndim == 2:
        return affine_transform(image, np.eye(2) * step, offset, output_shape=(size, size),
                                order=1, mode='nearest', prefilter=False)
    return np.stack([affine_transform(image[..., c], np.eye(2) * step, offset,
                    output_shape=(size, size), order=1, mode='nearest', prefilter=False)
                    for c in range(3)], axis=-1)


class Pairs(RawSet):
    def __init__(self, reg, rows, size, pack):
        super().__init__(np.asarray(reg['gidx'])[rows], pack=str(pack))
        self.hr = np.asarray(reg['hist_row'])[rows]
        self.scale = np.asarray(reg['scale'])[rows]
        self.dy, self.dx = np.asarray(reg['dy'])[rows], np.asarray(reg['dx'])[rows]
        self.rgb_scale, self.size, self.hist = float(reg['rgb_scale']), size, None

    def __getitem__(self, i):
        raw, phase, g = super().__getitem__(i)
        if self.hist is None:
            self.hist = np.load(self.pack + '/histology.npy', mmap_mode='r')
        y = registered_rgb(self.hist[int(self.hr[i])], self.scale[i], self.dy[i], self.dx[i], self.size)
        y = torch.from_numpy(np.clip(y / self.rgb_scale, 0, 1).transpose(2, 0, 1).copy())
        return raw, phase, g, y


def read_registration(path, targets, pack, holdout):
    with np.load(path, allow_pickle=False) as f:
        reg = {k: f[k] for k in f.files}
    meta = json.loads(str(reg['meta']))
    if meta['version'] != VERSION or meta['holdout'] != holdout or meta['targets_sha256'] != digest(targets):
        raise ValueError('registration version/holdout/targets mismatch; rerun vs_register')
    hc = np.load(Path(pack) / 'hist_cells.npy')
    if not np.array_equal(reg['gidx'], hc) or not np.array_equal(reg['hist_row'], np.arange(len(hc))):
        raise ValueError('registration does not match hist_cells order')
    with np.load(targets) as t:
        if not np.array_equal(t['hist_row'][hc], np.arange(len(hc))):
            raise ValueError('targets hist_row and hist_cells disagree')
    tr, te = split_rows(targets, reg['gidx'], holdout)
    tr, te = tr[reg['good'][tr]], te[reg['good'][te]]
    with np.load(targets) as t:   # red-team N028 M1: N003 area-outlier pairs never used for training (kept in test)
        tr = tr[~t['gs_pair_suspect'][reg['gidx'][tr]]]
    if not len(tr) or not len(te):
        raise ValueError('empty accepted training or test set')
    return reg, tr, te


def make_loader(dataset, bs, workers, sampler=None):
    kw = dict(batch_size=bs, shuffle=False) if sampler is None else dict(batch_sampler=sampler)
    return torch.utils.data.DataLoader(dataset, **kw, num_workers=workers,
        pin_memory=torch.cuda.is_available(), generator=torch.Generator().manual_seed(811),
        **(dict(prefetch_factor=2, persistent_workers=True) if workers else {}))


def normalize_batch(batch, norm, device):
    raw, phase, g, y = batch
    x = norm(raw.to(device, non_blocking=True), phase.to(device, non_blocking=True),
             g.to(device, non_blocking=True), ch='all', phase=True, shift=0)
    return x.contiguous(memory_format=torch.channels_last), y.to(device, non_blocking=True)


def translate_pair(x, y, radius=4):
    """Per-pair translation in LF pixels, scaled for 256px RGB; replicate padding."""
    if radius == 0:
        return x, y
    shifts = torch.randint(-radius, radius + 1, (len(x), 2)).tolist()
    def translate(z, factor):
        r = radius * factor
        p = F.pad(z, (r, r, r, r), mode='replicate')
        return torch.cat([p[i:i+1, :, r-dy*factor:r-dy*factor+z.shape[-2],
                                r-dx*factor:r-dx*factor+z.shape[-1]]
                          for i, (dy, dx) in enumerate(shifts)])
    return translate(x, 1), translate(y, y.shape[-1] // 128)


def block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU(),
                         nn.Conv2d(cout, cout, 3, padding=1), nn.GroupNorm(8, cout), nn.SiLU())


class UNet(nn.Module):
    def __init__(self, width=32, out_size=128):
        super().__init__()
        self.out_size = out_size
        self.enc = nn.ModuleList([block(24, width)] + [block(width * 2**i, width * 2**(i+1)) for i in range(3)])
        self.dec = nn.ModuleList([block(width * (2**i + 2**(i-1)), width * 2**(i-1)) for i in (3, 2, 1)])
        self.high = block(width, width) if out_size == 256 else nn.Identity()
        self.head = nn.Conv2d(width, 3, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(F.avg_pool2d(x, 2) if i else x)
            skips.append(x)
        for dec, skip in zip(self.dec, reversed(skips[:-1])):
            x = dec(torch.cat([F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False), skip], 1))
        if self.out_size == 256:
            x = F.interpolate(x, size=256, mode='bilinear', align_corners=False)
        return self.head(self.high(x)).sigmoid()


class PatchGAN(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(27, width, 4, 2, 1), nn.LeakyReLU(.2),
            nn.Conv2d(width, 2*width, 4, 2, 1), nn.GroupNorm(8, 2*width), nn.LeakyReLU(.2),
            nn.Conv2d(2*width, 4*width, 4, 2, 1), nn.GroupNorm(8, 4*width), nn.LeakyReLU(.2),
            nn.Conv2d(4*width, 1, 3, 1, 1))

    def forward(self, x, y):
        x = F.interpolate(x, size=y.shape[-2:], mode='bilinear', align_corners=False)
        return self.net(torch.cat([x, y], 1))


class Dino(nn.Module):
    def __init__(self, model, path):
        super().__init__()
        import timm
        self.vit = timm.create_model({'S': 'vit_small_patch14_dinov2', 'B': 'vit_base_patch14_dinov2'}[model],
                                     pretrained=False, num_classes=0, img_size=224)
        sd = torch.load(path, map_location='cpu', weights_only=True)['teacher']
        self.vit.load_state_dict({k[9:]: v for k, v in sd.items()
                                 if k.startswith('backbone.') and k != 'backbone.mask_token'}, strict=True)
        self.requires_grad_(False)
        self.eval()
        self.register_buffer('mean', torch.tensor([.485, .456, .406])[None, :, None, None])
        self.register_buffer('std', torch.tensor([.229, .224, .225])[None, :, None, None])

    def forward(self, x, patches=False):
        x = F.interpolate(x.float(), size=224, mode='bilinear', align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        if patches:
            return self.vit.forward_features(x)[:, 1:]
        return self.vit(x)


def robust_l1(pred, target):
    k = 5 if pred.shape[-1] == 128 else 9
    return .5 * F.l1_loss(pred.float(), target.float()) + .5 * F.l1_loss(
        F.avg_pool2d(pred.float(), k, 1, k//2, count_include_pad=False),
        F.avg_pool2d(target.float(), k, 1, k//2, count_include_pad=False))


def perceptual_loss(dino, pred, target):
    with torch.no_grad():
        truth = F.normalize(dino(target, patches=True).float(), dim=-1)
    generated = F.normalize(dino(pred, patches=True).float(), dim=-1)
    return (1 - (generated * truth).sum(-1)).mean()


def amp(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if torch.device(device).type == 'cuda' else contextlib.nullcontext()


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch'])
    if s['cuda']:
        torch.cuda.set_rng_state_all(s['cuda'])


class EpochBatches:
    def __init__(self, n, bs, seed):
        self.n, self.bs, self.seed, self.epoch, self.start = n, bs, seed, 0, 0

    @property
    def total(self):
        return (self.n + self.bs - 1) // self.bs

    def __len__(self):
        return self.total - self.start

    def __iter__(self):
        order = torch.randperm(self.n, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        for b in range(self.start, self.total):
            yield order[b*self.bs:(b+1)*self.bs]


def image_metrics(pred, real):
    """PSNR and Gaussian-window SSIM (11px, sigma=1.5), per image."""
    pred, real = pred.float(), real.float()
    mse = (pred - real).square().mean((1, 2, 3))
    psnr = -10 * mse.clamp_min(1e-12).log10()
    k = torch.arange(-5, 6, device=pred.device).float()
    k = torch.exp(-k.square() / (2 * 1.5**2)); k /= k.sum()
    kernel = (k[:, None] * k[None, :])[None, None].expand(3, 1, 11, 11)
    smooth = lambda z: F.conv2d(z, kernel, groups=3)
    ux, uy = smooth(pred), smooth(real)
    vx = (smooth(pred.square()) - ux.square()).clamp_min(0)
    vy = (smooth(real.square()) - uy.square()).clamp_min(0)
    cov = smooth(pred * real) - ux * uy
    ssim = ((2*ux*uy + .01**2) * (2*cov + .03**2) /
            ((ux.square()+uy.square()+.01**2) * (vx+vy+.03**2))).mean((1, 2, 3))
    return psnr, ssim


def retrieval(generated, real):
    scores = np.asarray(generated, np.float32) @ np.asarray(real, np.float32).T
    ranks = (scores >= scores.diagonal()[:, None] - 1e-7).sum(1)
    return dict(top1=float(np.mean(ranks <= 1)), top5=float(np.mean(ranks <= 5)),
                candidates=len(real), tie_tolerance=1e-7)
