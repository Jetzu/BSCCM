"""Extract a frozen SSL or AE representation while preserving the reference dump's row order."""
import argparse
import os
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from n087_image_ae import Encoder, GPUNorm, RawSet, W, normalise


def validate_rows(rows, flags, batches):
    if rows.ndim != 1 or rows.dtype.kind not in 'iu' or not len(rows):
        raise ValueError('G must be a non-empty vector of integer row IDs')
    if (rows.min() < 0 or rows.max() >= len(batches) or len(np.unique(rows)) != len(rows)
            or flags.shape != rows.shape or set(np.unique(flags)) != {0, 1}):
        raise ValueError('invalid reference row IDs or is_eval flags')
    source, held = np.unique(batches[rows[flags == 0]]), np.unique(batches[rows[flags == 1]])
    if source.size != 1 or held.size != 1 or {source.item(), held.item()} != {0, 1}:
        raise ValueError('expected opposite source/evaluation draws')
    return int(held.item())


def load_encoder(kind, path, held, batch, ssl_excluded_batch=None):
    if kind == 'ae':
        blob = torch.load(path, weights_only=True, map_location='cpu')
        expected = 'b1to0' if held == 0 else 'b0to1'
        if blob['protocol'] != expected:
            raise ValueError(f'AE protocol disagrees with reference rows (expected {expected})')
        model = Encoder(blob['dim'])
        model.load_state_dict(blob['model'])
        return model, blob['dim']

    import timm
    model = timm.create_model('resnet50', pretrained=False, in_chans=48, num_classes=0)
    if ssl_excluded_batch is not None and ssl_excluded_batch != held:
        raise ValueError('declared SSL exclusion disagrees with the reference rows')
    # The repo exports bare encoder weights. The sibling trainer checkpoint binds their
    # exact tensors to the source-row list; the full ckpt.pt can also be passed directly.
    proof_path = path.with_name('ckpt.pt')
    if proof_path.exists():
        proof = torch.load(proof_path, map_location='cpu', weights_only=False)
        trained = np.asarray(proof['rows'])
        if (proof['config']['arch'] != 'resnet50' or held not in proof['config']['exclude_batch']
                or trained.ndim != 1 or trained.dtype.kind not in 'iu' or not len(trained)
                or trained.min() < 0 or trained.max() >= len(batch)
                or not np.all(batch[trained] == 1 - held)):
            raise ValueError('SSL checkpoint provenance disagrees with reference draws')
        weights = proof['model']
        if path != proof_path:
            exported = torch.load(path, map_location='cpu', weights_only=True)
            if (exported.keys() != weights.keys()
                    or any(not torch.equal(v, weights[k]) for k, v in exported.items())):
                raise ValueError('SSL export does not match the source-verified checkpoint')
    else:
        if ssl_excluded_batch is None:
            raise ValueError('bare SSL export lacks sibling ckpt.pt; supply its documented --ssl-excluded-batch')
        weights = torch.load(path, map_location='cpu', weights_only=True)
    model.load_state_dict(weights)
    return model, model.num_features


@torch.inference_mode()
def extract(a):
    if a.out.resolve() in {a.rows_from.resolve(), a.ckpt.resolve()}:
        raise ValueError('output must differ from the reference and checkpoint')
    with np.load(a.rows_from) as ref:
        rows, flags, names = (ref[key] for key in ('G', 'is_eval', 'names'))
    with np.load(Path(W) / 'targets_v3.npz') as targets:
        batches = targets['batch']
    held = validate_rows(rows, flags, batches)
    model, dim = load_encoder(a.encoder, a.ckpt, held, batches, a.ssl_excluded_batch)
    model.eval().to(a.device, memory_format=torch.channels_last)
    features = np.empty((len(rows), dim), np.float16)
    norm = GPUNorm(a.device)
    dl = DataLoader(RawSet(rows), batch_size=a.bs, num_workers=a.workers,
                    pin_memory=a.device == 'cuda')
    start = 0
    for sample in dl:
        x = normalise(norm, sample, a.device)
        if a.encoder == 'ssl':
            x = torch.cat((x, torch.ones_like(x)), 1)
        else:
            x = torch.where(torch.isfinite(x), x, 0.)
        with torch.autocast(a.device, dtype=torch.bfloat16, enabled=a.device == 'cuda'):
            h = model(x.contiguous(memory_format=torch.channels_last))
        block = h.to(dtype=torch.float16, device='cpu').numpy()
        end = start + len(x)
        if not np.array_equal(rows[start:end], sample[2].numpy()):
            raise RuntimeError('image loader changed the row order')
        if not np.isfinite(block).all():
            raise FloatingPointError(f'nonfinite extracted features at offsets [{start}, {end})')
        features[start:end] = block
        start = end
    if start != len(rows):
        raise RuntimeError('incomplete extraction')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_name(a.out.name + '.tmp')
    with tmp.open('wb') as f:
        np.savez(f, H=features, G=rows, is_eval=flags, names=names)
    os.replace(tmp, a.out)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--encoder', choices=('ssl', 'ae'), required=True)
    p.add_argument('--ckpt', type=Path, required=True)
    p.add_argument('--rows-from', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda'), default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--bs', type=int, default=128)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--ssl-excluded-batch', type=int, choices=(0, 1),
                   help='attest the training exclusion for a standalone legacy SSL export')
    a = p.parse_args(argv)
    if a.bs < 1 or a.workers < 0:
        p.error('invalid loader options')
    return a


if __name__ == '__main__':
    extract(parse_args())
