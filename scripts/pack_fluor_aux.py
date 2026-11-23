#!/usr/bin/env python3
"""Pack draw-local, corner-subtracted dense fluorescence targets; resumable CPU job."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root (scripts/ -> BSCCM/)
from bsccm.dense_aux import CHANNELS, CONDITIONS, disk, pooled_crop, row_digest

W = Path('/hnvme/workspace/v181bb10-bsccm')
STOP = False
FLUOR = None
PARAMS = None
BATCH = None
SAMPLE = np.flatnonzero(disk().ravel())[np.linspace(0, disk().sum()-1, 64, dtype=int)]


def stop_handler(signum, frame):
    global STOP
    STOP = True
    print(f'signal {signum}: finish durable chunk, exit 143', flush=True)


def check_stop():
    if STOP:
        raise SystemExit(143)


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def init_worker(path, params, batch):
    global FLUOR, PARAMS, BATCH
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGUSR1, signal.SIG_IGN)
    FLUOR = np.load(path, mmap_mode='r')
    PARAMS, BATCH = params, batch


def sample_job(rows):
    return pooled_crop(FLUOR[rows]).reshape(len(rows), 6, -1)[:, :, SAMPLE]


def asinh_rows(rows):
    x = pooled_crop(FLUOR[rows])
    for draw in np.unique(BATCH[rows]):
        sel = BATCH[rows] == draw
        cof = np.asarray(PARAMS[str(int(draw))]['cofactor'], np.float32)
        x[sel] = np.arcsinh(x[sel] / cof[None, :, None, None])
    return x


def moments_job(rows):
    x = asinh_rows(rows).astype(np.float64)
    return dict(count=len(rows)*64*64, sum=x.sum((0, 2, 3)).tolist(),
                sumsq=np.square(x).sum((0, 2, 3)).tolist())


def write_job(rows):
    x = asinh_rows(rows)
    for draw in np.unique(BATCH[rows]):
        sel = BATCH[rows] == draw
        p = PARAMS[str(int(draw))]
        mean, sd = np.asarray(p['mean'], np.float32), np.asarray(p['std'], np.float32)
        x[sel] = (x[sel] - mean[None, :, None, None]) / sd[None, :, None, None]
    if not np.isfinite(x).all() or np.max(np.abs(x)) > np.finfo(np.float16).max:
        raise ValueError('non-finite/unrepresentable float16 fluorescence target')
    return x.astype(np.float16)


def jobs_in_order(function, chunks, workers, pack, params, batch):
    """Bounded results (<=2*workers chunks); workers never create output files."""
    chunks = iter(chunks)
    if workers == 1:
        init_worker(str(pack), params, batch)
        # init_worker is also used in the parent in serial mode: restore checkpoint handlers.
        signal.signal(signal.SIGTERM, stop_handler)
        signal.signal(signal.SIGUSR1, stop_handler)
        for rows in chunks:
            check_stop()
            yield function(rows)
        return
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork'), initializer=init_worker,
                             initargs=(str(pack), params, batch)) as pool:
        pending = []
        for _ in range(2*workers):
            rows = next(chunks, None)
            if rows is None:
                break
            pending.append(pool.submit(function, rows))
        try:
            while pending:
                check_stop()
                future = pending.pop(0)
                yield future.result()
                rows = next(chunks, None)
                if rows is not None and not STOP:
                    pending.append(pool.submit(function, rows))
        finally:
            for future in pending:
                future.cancel()


def chunks_of(rows, size, start=0):
    return [rows[i:i+size] for i in range(start, len(rows), size)]


def file_id(path):
    st = path.stat()
    return dict(path=str(path.resolve()), size=st.st_size, mtime_ns=st.st_mtime_ns)


def pack(args):
    source, out = args.source, args.out
    pack_path, index_path = source/'pack/fluor.npy', source/'bsccm_index.parquet'
    targets_path = source/args.targets
    fluor = np.load(pack_path, mmap_mode='r')
    idx = pd.read_parquet(index_path)
    with np.load(targets_path) as t:
        batch, split, hist_row = t['batch'], t['split'], t['hist_row']
    n = len(batch)
    if fluor.shape != (n, 6, 128, 128) or fluor.dtype != np.uint16:
        raise ValueError('expected native uint16 (N_all,6,128,128) fluorescence pack')
    if (not np.array_equal(idx.global_index, np.arange(n)) or not np.array_equal(idx.batch, batch)):
        raise ValueError('global index / draw mismatch')
    if (args.smoke and not 1 <= n <= 64) or (not args.smoke and n != 412941):
        raise ValueError('production requires ALL 412941 rows; --smoke requires <=64 fixture rows')
    if set(idx.antibodies) - set(CONDITIONS) or set(np.unique(batch)) != {0, 1}:
        raise ValueError('unexpected antibody/draw schema')
    manifest = dict(version=1, shape=[n, 6, 64, 64], dtype='float16',
                    row_sha256=row_digest(batch, split, idx.antibodies), channels=list(CHANNELS),
                    inputs=[file_id(p) for p in (pack_path, index_path, targets_path)],
                    code_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), Path(__file__).resolve().parent.parent/'bsccm/dense_aux.py')],
                    chunk_size=args.chunk_size, cofactor_cells=args.cofactor_cells,
                    cofactor_pixels=SAMPLE.tolist(), smoke=args.smoke)
    meta_path = out.with_suffix('.json')
    if meta_path.exists():
        state = json.loads(meta_path.read_text())
        if state['manifest'] != manifest:
            raise ValueError('different code/inputs/options: choose a new output; refusing to overwrite')
    else:
        if out.exists():
            raise ValueError('output exists without metadata; refusing to overwrite')
        state = dict(manifest=manifest, row_sha256=manifest['row_sha256'], channels=list(CHANNELS),
                     complete=False, draws={}, standardize={}, next_row=0,
                     method=dict(background='per-cell/channel median outside r<64 disk centred at (64,64)',
                                 pool='2x2 arithmetic mean before asinh; native channel order reversed',
                                 cofactor='max(1 ADU, 1.4826*MAD) on 64 fixed foreground pixels per selected split-0 unstained cell',
                                 selection='up to cofactor_cells evenly spaced split-0 unstained global rows PER DRAW, Giemsa-paired (hist_row>=0) excluded',
                                 standardize='population mean/std of ALL pixels of ALL split-0 non-Giemsa-paired cells PER DRAW; sd floor 1e-6'))
        atomic_json(meta_path, state)
    if state['complete']:
        test = np.load(out, mmap_mode='r')
        if test.shape != tuple(manifest['shape']) or test.dtype != np.float16:
            raise ValueError('completed pack header mismatch')
        print('already complete; identical manifest', flush=True)
        return
    for draw in (0, 1):
        key = str(draw)
        if key in state['draws']:
            continue
        rows = np.flatnonzero((batch == draw) & (split == 0) & (hist_row < 0) & (idx.antibodies.to_numpy() == 'unstained'))   # N074 red-team MINOR-1: no Giemsa-paired cells
        if not len(rows):
            raise ValueError(f'no split-0 unstained cells for draw {draw}')
        rows = rows[np.linspace(0, len(rows)-1, min(args.cofactor_cells, len(rows)), dtype=int)]
        parts = list(jobs_in_order(sample_job, chunks_of(rows, args.chunk_size), args.workers,
                                   pack_path, {}, batch))
        sample = np.concatenate(parts).transpose(1, 0, 2).reshape(6, -1)
        median = np.median(sample, axis=1)
        cofactor = np.maximum(1., 1.4826*np.median(np.abs(sample-median[:, None]), axis=1))
        state['draws'][key] = dict(cofactor=cofactor.tolist(), cofactor_n_cells=len(rows),
                                  cofactor_rows_sha256=hashlib.sha256(rows.astype('<i8').tobytes()).hexdigest())
        atomic_json(meta_path, state)
        check_stop()
    for draw in (0, 1):
        key = str(draw)
        if 'mean' in state['draws'][key]:
            continue
        rows = np.flatnonzero((batch == draw) & (split == 0) & (hist_row < 0))   # N074 red-team MINOR-1
        if not len(rows):
            raise ValueError(f'no split-0 cells for draw {draw}')
        progress = state['standardize'].setdefault(key, dict(next=0, count=0, sum=[0.]*6, sumsq=[0.]*6))
        chunks = chunks_of(rows, args.chunk_size, progress['next'])
        for rr, moments in zip(chunks, jobs_in_order(moments_job, chunks, args.workers,
                                                    pack_path, state['draws'], batch)):
            progress['next'] += len(rr)
            progress['count'] += moments['count']
            for field in ('sum', 'sumsq'):
                progress[field] = (np.asarray(progress[field])+moments[field]).tolist()
            atomic_json(meta_path, state)
            check_stop()
        mean = np.asarray(progress['sum']) / progress['count']
        var = np.asarray(progress['sumsq']) / progress['count'] - mean**2
        state['draws'][key].update(mean=mean.tolist(), std=np.maximum(np.sqrt(np.maximum(var, 0)), 1e-6).tolist(),
                                   standardize_n_cells=len(rows))
        atomic_json(meta_path, state)
    check_stop()
    if out.exists():
        output = np.load(out, mmap_mode='r+')
        if output.shape != tuple(manifest['shape']) or output.dtype != np.float16:
            raise ValueError('partial pack header mismatch')
    else:
        if state['next_row']:
            raise ValueError('missing partial output with committed rows')
        output = np.lib.format.open_memmap(out, mode='w+', dtype=np.float16, shape=tuple(manifest['shape']))
    chunks = chunks_of(np.arange(n), args.chunk_size, state['next_row'])
    with out.open('rb') as durable:
        for rows, image in zip(chunks, jobs_in_order(write_job, chunks, args.workers,
                                                    pack_path, state['draws'], batch)):
            output[rows] = image
            output.flush()
            os.fsync(durable.fileno())
            # Commit progress ONLY after its data are durable. Interrupted chunks are overwritten.
            state['next_row'] = int(rows[-1])+1
            atomic_json(meta_path, state)
            print(f'committed {state["next_row"]}/{n}', flush=True)
            check_stop()
    state['complete'] = True
    atomic_json(meta_path, state)
    print('complete', str(out), output.shape, output.dtype, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=W)
    p.add_argument('--out', type=Path, default=W/'fluor_aux64.npy')
    p.add_argument('--targets', default='targets_v5cx_v10.npz')
    p.add_argument('--workers', type=int, default=48)
    p.add_argument('--chunk-size', type=int, default=512)   # N074 red-team MINOR-5: fewer flush/fsync rounds
    p.add_argument('--cofactor-cells', type=int, default=4096)
    p.add_argument('--smoke', action='store_true', help='explicitly allow a <=64-row fixture source')
    args = p.parse_args()
    if min(args.workers, args.chunk_size, args.cofactor_cells) < 1 or args.out.suffix != '.npy':
        p.error('positive workers/chunk-size/cofactor-cells and a .npy output required')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    # Lock an already required metadata inode's directory (no extra lock/chunk files).
    # Directory lock serializes packers targeting this directory, including resume chains.
    lock = os.open(args.out.parent, os.O_RDONLY)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pack(args)
    finally:
        os.close(lock)


if __name__ == '__main__':
    main()
