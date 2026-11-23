"""N074 training-only fluorescence supervision. No fluorescence enters the LF encoder."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

CHANNELS = ('426-446', '500-550', '550-570', '585-625', '627-673', '690-')
CONDITIONS = ('unstained', 'all', 'CD45', 'CD123', 'CD19', 'CD56', 'CD14', 'CD16', 'HLA-DR', 'CD3')


def row_digest(batch, split, antibodies):
    h = hashlib.sha256()
    for arr in (np.asarray(batch, '<i8'), np.asarray(split, '<i8')):
        h.update(arr.tobytes())
    h.update(json.dumps(list(map(str, antibodies)), separators=(',', ':')).encode())
    return h.hexdigest()


def disk(size=64):
    y, x = np.indices((size, size))
    return (y - size // 2)**2 + (x - size // 2)**2 < (size // 2)**2


def pooled_crop(raw):
    """Native pack channels 5..0 -> CHANNELS; signed corner subtraction before pooling."""
    x = np.asarray(raw, np.float32)[:, ::-1].copy()
    x -= np.median(x[..., ~disk(128)], axis=-1)[..., None, None]
    return x.reshape(len(x), 6, 64, 2, 64, 2).mean(axis=(3, 5))


def roll_batch(x, shifts):
    """Same integer cyclic translation as GPUNorm, in 128-pixel coordinates."""
    return torch.stack([torch.roll(v, (int(dy), int(dx)), (-2, -1))
                        for v, (dy, dx) in zip(x, shifts)])


def reflect(x):
    return x.flip(-1).flip(-2).transpose(-1, -2)


def align_target(target, shifts, selected, zoom=1):
    """Nearest 64->128, exact shared roll/crop/reflection, mean 128->64.

    Odd LF shifts are half a target pixel: this implements their exact area overlap
    for the piecewise-constant 64-pixel target. No spectral channel permutation.
    Return the identically warped loss mask as well as the target.
    """
    weights = torch.as_tensor(np.where(disk(), 1., .25), device=target.device, dtype=target.dtype)
    weights = weights[None, None].expand(len(target), 1, -1, -1)
    both = F.interpolate(torch.cat([target, weights], 1), scale_factor=2, mode='nearest')
    both = roll_batch(both, shifts)
    if zoom > 1:
        size = 128 // zoom
        off = (128 - size) // 2
        both = F.interpolate(both[..., off:off + size, off:off + size], (128, 128),
                             mode='bilinear', align_corners=False)
    if selected.any():
        both[selected] = reflect(both[selected])
    both = F.avg_pool2d(both, 2)
    return both[:, :6], both[:, 6:]


def pooled_with_spatial(encoder, x):
    """Capture timm.forward_features while calling the original encoder.forward.

    LFBackbone's learned LED gate/resize, timm.forward_head, pooling, dropout and
    stochastic depth all run exactly once. The temporary Python method is restored
    even on failure; it is never a module/parameter/checkpoint key.
    """
    backbone = encoder.net if hasattr(encoder, 'gate') and hasattr(encoder, 'net') else encoder
    if not hasattr(backbone, 'forward_features'):
        raise ValueError('--dense-aux requires an encoder with timm.forward_features spatial output')
    original = backbone.forward_features
    had_override = 'forward_features' in backbone.__dict__
    saved = backbone.__dict__.get('forward_features')
    features = []

    def capture(*args, **kwargs):
        z = original(*args, **kwargs)
        if not isinstance(z, torch.Tensor) or z.ndim != 4 or z.shape[1] != encoder.num_features:
            raise ValueError('--dense-aux requires an NCHW pre-pool spatial map; token-only encoders are unsupported')
        features.append(z)
        return z

    backbone.forward_features = capture
    try:
        h = encoder(x)
    finally:
        if had_override:
            backbone.forward_features = saved
        else:
            del backbone.forward_features
    if len(features) != 1:
        raise ValueError('--dense-aux expected exactly one spatial encoder call')
    return h, features[0]


class DenseDecoder(nn.Module):
    def __init__(self, features, width=64):
        super().__init__()
        self.project = nn.Conv2d(features, width, 1)
        self.condition = nn.Embedding(len(CONDITIONS), 16)
        self.film = nn.Linear(16, width * 2)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Conv2d(width, width, 3, padding=1), nn.GELU())
                                     for _ in range(4)])
        self.output = nn.Conv2d(width, 6, 1)

    def forward(self, spatial, condition):
        z = self.project(spatial.float())
        scale, bias = self.film(self.condition(condition)).chunk(2, dim=1)
        z = F.gelu(z * (1 + scale[..., None, None]) + bias[..., None, None])
        for block in self.blocks:
            size = min(64, z.shape[-1] * 2)
            z = block(F.interpolate(z, (size, size), mode='bilinear', align_corners=False))
        return self.output(F.interpolate(z, (64, 64), mode='bilinear', align_corners=False))


def dense_mse(prediction, target, weights):
    return ((prediction.float() - target).square() * weights).sum() / (weights.sum() * 6)


class DenseTargets:
    """Only permitted training-row indices can read fluorescence or panel IDs."""
    def __init__(self, path, train_rows, batch, split, index_path):
        import pandas as pd
        self.path = Path(path)
        self.meta = json.loads(self.path.with_suffix('.json').read_text())
        if self.meta.get('complete') is not True:
            raise ValueError('--dense-aux pack is incomplete')
        self.images = np.load(self.path, mmap_mode='r')
        if self.images.dtype != np.float16 or self.images.shape != (len(batch), 6, 64, 64):
            raise ValueError('--dense-aux expects float16 (N_all,6,64,64) indexed by global row')
        idx = pd.read_parquet(index_path)
        if (not np.array_equal(idx.global_index, np.arange(len(batch))) or
                not np.array_equal(idx.batch, batch)):
            raise ValueError('--dense-aux index/global row/draw mismatch')
        digest = row_digest(batch, split, idx.antibodies)
        if self.meta.get('row_sha256') != digest or self.meta.get('channels') != list(CHANNELS):
            raise ValueError('--dense-aux pack row/split/panel/channel schema mismatch')
        if set(idx.antibodies) - set(CONDITIONS):
            raise ValueError('--dense-aux unknown antibody condition')
        self.panels = np.array([CONDITIONS.index(v) for v in idx.antibodies], np.int64)
        self.allowed = np.zeros(len(batch), bool)
        self.allowed[train_rows] = True
        self.batch = np.asarray(batch)
        draws = np.unique(self.batch[train_rows])
        if len(draws) != 1:
            raise ValueError('--dense-aux requires one training draw')
        self.draw = draws[0]
        # Pin both transform constants and the completed pack's identity on resume.
        stat = self.path.stat()
        self.identity = dict(metadata=self.meta, path=str(self.path.resolve()),
                             size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                             conditions=list(CONDITIONS))

    def get(self, rows, device):
        rows = rows.detach().cpu().numpy().astype(np.int64, copy=False)
        if (rows.ndim != 1 or (rows < 0).any() or (rows >= len(self.allowed)).any() or
                not self.allowed[rows].all() or not (self.batch[rows] == self.draw).all()):
            raise AssertionError('held-out draw/non-training row reached the dense decoder loss')
        target = torch.from_numpy(np.array(self.images[rows], dtype=np.float32)).to(device)
        panel = torch.from_numpy(self.panels[rows]).to(device)
        if not torch.isfinite(target).all():
            raise ValueError('non-finite fluorescence target')
        return target, panel
