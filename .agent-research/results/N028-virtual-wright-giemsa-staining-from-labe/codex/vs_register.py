"""CPU global GS scale fit and per-pair phase-correlation registration."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import argparse
import json
import os
import signal
import time
# Parallelism is across processes; one native thread per registration worker.
for _var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[_var] = "1"
import multiprocessing as mp
import numpy as np
from scipy.ndimage import gaussian_filter
from vs_common import W, VERSION, atomic_npz, digest, registered_rgb, split_rows

_H = _L = _HC = None


def init_worker(pack):
    global _H, _L, _HC
    _H = np.load(Path(pack) / 'histology.npy', mmap_mode='r')
    _L = np.load(Path(pack) / 'led_array.npy', mmap_mode='r')
    _HC = np.load(Path(pack) / 'hist_cells.npy')


def ncc(a, b, dy, dx, border=8):
    y0, y1 = max(border, dy+border), min(128-border, 128+dy-border)
    x0, x1 = max(border, dx+border), min(128-border, 128+dx-border)
    if y1 <= y0 or x1 <= x0:
        return -1.
    a = a[y0:y1, x0:x1].ravel().astype(np.float64)
    b = b[y0-dy:y1-dy, x0-dx:x1-dx].ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den > 1e-8 else -1.


def phase_shift(a, b, max_shift=12):
    """Translation to apply to b, in LF pixels: phase peak + local NCC refinement."""
    a, b = gaussian_filter(a.astype(np.float32), 1), gaussian_filter(b.astype(np.float32), 1)
    if a.std() < 1e-6 or b.std() < 1e-6:
        return 0, 0, -1., 0.
    win = np.outer(np.hanning(128), np.hanning(128))
    fa = np.fft.fft2((a-a.mean()) * win)
    fb = np.fft.fft2((b-b.mean()) * win)
    cross = fa * fb.conj()
    corr = np.fft.fftshift(np.fft.ifft2(cross / np.maximum(np.abs(cross), 1e-8)).real)
    sub = corr[64-max_shift:65+max_shift, 64-max_shift:65+max_shift]
    py, px = np.unravel_index(sub.argmax(), sub.shape)
    dy, dx = py-max_shift, px-max_shift
    candidates = [(ncc(a, b, y, x), y, x) for y in range(max(-max_shift, dy-2), min(max_shift, dy+2)+1)
                   for x in range(max(-max_shift, dx-2), min(max_shift, dx+2)+1)]
    peak, dy, dx = max(candidates)
    return dy, dx, peak, float(sub.max())


def fit_one(task):
    r, scales, max_shift = task
    lf = np.asarray(_L[int(_HC[r]), 0], np.float32)
    grey = np.asarray(_H[r], np.float32).mean(-1)
    scores = [phase_shift(lf, registered_rgb(grey, s), max_shift)[2] for s in scales]
    return scores, np.asarray(_H[r, ::8, ::8], np.float32).ravel()


def register_one(task):
    r, scale, max_shift = task
    lf = np.asarray(_L[int(_HC[r]), 0], np.float32)
    grey = np.asarray(_H[r], np.float32).mean(-1)
    return phase_shift(lf, registered_rgb(grey, scale), max_shift)


def register_indexed(task):
    return task[0], register_one(task)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--pack', default=W+'/pack')
    p.add_argument('--targets', default=W+'/targets_v6.npz')
    p.add_argument('--holdout', choices=['fov', 'batch0', 'batch1'], default='fov')
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--fit-samples', type=int, default=256)
    p.add_argument('--scale-min', type=float, default=.6)
    p.add_argument('--scale-max', type=float, default=.8)
    p.add_argument('--scale-steps', type=int, default=21)
    p.add_argument('--max-shift', type=int, default=12)
    p.add_argument('--min-peak', type=float, default=.4, help='positive overlap NCC, not phase height')
    p.add_argument('--seed', type=int, default=0)
    a = p.parse_args()
    if not (0 < a.scale_min < a.scale_max <= 1 and a.scale_steps >= 2 and 2 <= a.max_shift <= 24
            and a.workers > 0 and a.fit_samples > 0 and -1 < a.min_peak < 1):
        p.error('invalid registration arguments')
    stop = False
    def on_signal(signum, frame):
        nonlocal stop
        stop = True
    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, on_signal)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    state_path = Path(str(out)+'.state.npz')
    config = {k: v for k, v in vars(a).items() if k not in ['out', 'workers']}
    config.update(version=VERSION, targets_sha256=digest(a.targets),
                  hist_cells_sha256=digest(Path(a.pack)/'hist_cells.npy'))
    if out.exists():
        with np.load(out) as f:
            old = json.loads(str(f['meta']))
        if old['registration_config'] != config:
            raise ValueError('existing registration config mismatch; use a fresh --out')
        print(json.dumps(dict(event='already_complete', **old['report'])), flush=True)
        return
    hc = np.load(Path(a.pack)/'hist_cells.npy')
    with np.load(a.targets) as t:
        if not np.array_equal(t['hist_row'][hc], np.arange(len(hc))):
            raise ValueError('targets hist_row != hist_cells mapping')
    tr, te = split_rows(a.targets, hc, a.holdout)
    if not len(tr) or not len(te):
        raise ValueError('empty split')
    fit_rows = np.random.default_rng(a.seed).choice(tr, min(a.fit_samples, len(tr)), replace=False)
    scales = np.linspace(a.scale_min, a.scale_max, a.scale_steps)
    print(json.dumps(dict(placement='CPU', workers=a.workers,
          reason='independent FFT/resize tasks; no GPU advantage', fit_cells=len(fit_rows))), flush=True)
    with mp.get_context('spawn').Pool(a.workers, initializer=init_worker, initargs=(a.pack,)) as pool:
        if state_path.exists():
            with np.load(state_path) as state:
                if json.loads(str(state['config'])) != config:
                    raise ValueError('registration resume config mismatch')
                scale, rgb_scale = float(state['scale']), float(state['rgb_scale'])
                med, results = state['med'], state['results']
        else:
            fit = []
            for result in pool.imap(fit_one, [(int(r), scales, a.max_shift) for r in fit_rows], chunksize=1):
                fit.append(result)
                if stop:
                    print(json.dumps(dict(event='stopped', stage='scale_fit', restart_stage=True)), flush=True)
                    return
            scores = np.asarray([f[0] for f in fit])
            med = np.median(scores, axis=0)
            scale = float(scales[med.argmax()])
            rgb_scale = float(np.percentile(np.concatenate([f[1] for f in fit]), 99.9))
            if rgb_scale <= 0:
                raise ValueError('zero RGB training scale')
            results = np.full((len(hc), 4), np.nan, np.float64)
        def save_progress():
            atomic_npz(state_path, config=json.dumps(config), scale=scale, rgb_scale=rgb_scale,
                       med=med, results=results)
        save_progress()
        print(json.dumps(dict(global_scale=scale, scale_grid=scales.tolist(), median_ncc=med.tolist(),
                              rgb_scale=rgb_scale, completed=int(np.isfinite(results[:, 2]).sum()))), flush=True)
        remaining = np.flatnonzero(~np.isfinite(results[:, 2]))
        last_save = time.monotonic()
        for r, result in pool.imap_unordered(register_indexed,
                [(int(r), scale, a.max_shift) for r in remaining], chunksize=8):
            results[r] = result
            if stop or time.monotonic()-last_save > 300:
                save_progress(); last_save = time.monotonic()
            if stop:
                print(json.dumps(dict(event='stopped', stage='pairs',
                      completed=int(np.isfinite(results[:, 2]).sum()))), flush=True)
                return
    if not np.isfinite(results).all():
        raise RuntimeError('non-finite registration results')
    dy, dx, peak, phase_peak = results.T
    good = (peak >= a.min_peak) & (np.abs(dy) < a.max_shift) & (np.abs(dx) < a.max_shift)
    report = dict(n_total=len(hc), n_good=int(good.sum()), n_dropped=int((~good).sum()),
                  train_good=int(good[tr].sum()), test_good=int(good[te].sum()),
                  global_scale=scale, scale_at_boundary=bool(scale in (scales[0], scales[-1])),
                  median_peak=float(np.median(peak)), min_peak=a.min_peak)
    meta = dict(version=VERSION, holdout=a.holdout, targets_sha256=digest(a.targets),
                args=vars(a), report=report, registration_config=config, shift_convention='translation applied to GS at 128px')
    atomic_npz(out, gidx=hc, hist_row=np.arange(len(hc)), scale=np.full(len(hc), scale, np.float32),
               dy=dy.astype(np.int16), dx=dx.astype(np.int16), peak=peak.astype(np.float32),
               phase_peak=phase_peak.astype(np.float32), good=good, rgb_scale=rgb_scale,
               fit_rows=fit_rows, scale_grid=scales, fit_median_ncc=med, meta=json.dumps(meta))
    state_path.unlink(missing_ok=True)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
