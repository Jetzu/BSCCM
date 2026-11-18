"""Real-data, compact CPU fixture runner; production evaluate() is never replaced.

Usage: smoke_cpu.py --prepare ROOT | --root ROOT --script FILE -- <harness args>
Only fixture paths in imported bsccm modules are rebound. No source files change.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys
import numpy as np

REPO = '/home/hpc/v181bb/v181bb10/BSCCM'
DATA = '/hnvme/workspace/v181bb10-bsccm'
ap = argparse.ArgumentParser()
ap.add_argument('--prepare')
ap.add_argument('--root')
ap.add_argument('--script', default=str(Path(__file__).with_name('train_lf.py')))
a, rest = ap.parse_known_args()
if a.prepare:
    root = Path(a.prepare); (root / 'pack').mkdir(parents=True, exist_ok=True)
    t = dict(np.load(DATA + '/targets_v5cx_v10.npz'))
    has = (t['hist_row'] < 0) & (t['mask'].any(1) | (t['cls3'] >= 0))
    rows = []
    for slide in np.unique(t['slide'][has]):
        for split, count in ((0, 2), (2, 1)):
            rr = np.flatnonzero(has & (t['slide'] == slide) & (t['split'] == split))
            fovs, counts = np.unique(t['fov'][rr], return_counts=True)
            eligible = fovs[counts >= 8][:count]
            for f in eligible: rows.extend(rr[t['fov'][rr] == f][:8].tolist())
    rows = np.array(sorted(set(rows)), dtype=np.int64)
    n = len(t['Y'])
    np.savez(root / 'targets_v5cx_v10.npz', **{k: v[rows] if v.ndim and len(v) == n else v for k, v in t.items()})
    _meta = np.load(DATA + '/cellmeta.npz'); meta = {k: _meta[k] for k in ('pos_y', 'pos_x', 'slide')}
    np.savez(root / 'cellmeta.npz', **{k: v[rows] if v.ndim and len(v) == n else v for k, v in meta.items()})
    np.save(root / 'original_rows.npy', rows)
    for name in ('led_array', 'dpc'):
        src = np.load(DATA + '/pack/' + name + '.npy', mmap_mode='r')
        dst = np.lib.format.open_memmap(root / 'pack' / (name + '.npy'), mode='w+', dtype=src.dtype, shape=(len(rows),) + src.shape[1:])
        for i, g in enumerate(rows): dst[i] = src[g]
        dst.flush()
    for name in ('backgrounds_p50.npy', 'led_norm.npz'):
        dst = root / name
        if not dst.exists(): dst.symlink_to(Path(DATA) / name)
    print(f'Prepared {len(rows)} real cells, source split 0 / evaluation split 2, original IDs in original_rows.npy', flush=True)
    sys.exit(0)
if not a.root: ap.error('--root required')
sys.path.insert(0, REPO)
import torch
import bsccm.data as data
import bsccm.gpu_norm as norm
torch.set_num_threads(2)
data.W = norm.W = str(Path(a.root).resolve())
norm.RawSet.__init__.__defaults__ = (None, data.W + '/pack')
os.environ['WANDB_MODE'] = 'disabled'
sys.argv = [a.script] + (rest[1:] if rest[:1] == ['--'] else rest)
runpy.run_path(a.script, run_name='__main__')
