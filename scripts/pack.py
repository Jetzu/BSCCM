#!/usr/bin/env python3
"""Stage 2: repack BSCCM zarr-in-zip into fixed-record .npy arrays (np.load(mmap_mode='r') friendly), parallel + resumable.
usage: pack.py <BSCCM.zarr.zip> <index.npz> <zarrmeta.json> <outdir> [nproc]
Outputs (row = global_index, 0..412940; histology rows follow hist_cells.npy):
  led_array.npy (N,23,128,128) uint16 | fluor.npy (N,6,128,128) uint16 | dpc.npy (N,128,128) float32 | histology.npy (Nh,398,398,3) uint16"""
import sys, os, json, struct, time
import numpy as np, numcodecs
from multiprocessing import Pool
zp, ix, mj, out = sys.argv[1:5]; nproc = int(sys.argv[5]) if len(sys.argv) > 5 else 48
N = 412941; BLK = 256
_z = np.load(ix); I = {k: _z[k] for k in _z.files}; meta = json.load(open(mj)); EX = int(I['local_extra'])
def cfg(mod):
    assert len(meta[mod]) == 1, (mod, len(meta[mod])); m = meta[mod][0]; return m
M = {k: cfg(k) for k in ('led_array', 'fluor', 'dpc', 'histology')}
codec = {k: numcodecs.get_codec(M[k]['compressor']) for k in M}
assert all(M[k]['order'] == 'C' for k in M)
fill = {k: (M[k]['fill_value'] or 0) for k in M}
spec = {'led_array': ((N, 23, 128, 128), 'u2'), 'fluor': ((N, 6, 128, 128), 'u2'), 'dpc': ((N, 128, 128), 'f4'),
        'histology': ((len(I['h_cells']), 398, 398, 3), M['histology']['dtype'])}
assert M['led_array']['dtype'] == '<u2' and M['dpc']['dtype'] == '<f4' and M['histology']['chunks'] == [199, 199, 3]
os.makedirs(out, exist_ok=True)
if __name__ == '__main__' and not os.path.exists(out + '/.created'):
    for k, (sh, dt) in spec.items():
        np.lib.format.open_memmap(f'{out}/{k}.npy', mode='w+', dtype=np.dtype(dt), shape=sh).flush()
    np.save(out + '/hist_cells.npy', I['h_cells']); open(out + '/.created', 'w').write('ok')
_fd = None
def rd(off, size, name_hint=None):
    # data offset = header_offset + 30 + len(name) + extra; read local header to get the name length (exact)
    h = os.pread(_fd, 30, int(off)); assert h[:4] == b'PK\x03\x04'
    fn, ex = struct.unpack('<HH', h[26:30])
    return os.pread(_fd, int(size), int(off) + 30 + fn + ex)
def dec(mod, buf, nelem, dt):
    a = np.frombuffer(codec[mod].decode(buf), dtype=dt); assert a.size == nelem, (mod, a.size); return a
def work(b):
    global _fd
    if _fd is None: _fd = os.open(zp, os.O_RDONLY)
    g0, g1 = b * BLK, min(N, (b + 1) * BLK); miss = 0; n = g1 - g0
    led = np.full((n, 23, 128, 128), fill['led_array'], '<u2'); fl = np.full((n, 6, 128, 128), fill['fluor'], '<u2'); dp = np.full((n, 128, 128), fill['dpc'], '<f4')
    for i in range(n):
        g = g0 + i
        for c in range(23):
            s = I['led_sz'][g, c]
            if s: led[i, c] = dec('led_array', rd(I['led_off'][g, c], s), 128 * 128, '<u2').reshape(128, 128)
            else: miss += 1
        for c in range(6):
            s = I['fl_sz'][g, c]
            if s: fl[i, c] = dec('fluor', rd(I['fl_off'][g, c], s), 128 * 128, '<u2').reshape(128, 128)
            else: miss += 1
        s = I['dpc_sz'][g, 0]
        if s: dp[i] = dec('dpc', rd(I['dpc_off'][g, 0], s), 128 * 128, '<f4').reshape(128, 128)
        else: miss += 1
    for k, a in (('led_array', led), ('fluor', fl), ('dpc', dp)):
        m = np.load(f'{out}/{k}.npy', mmap_mode='r+'); m[g0:g1] = a; m.flush(); del m
    return b, miss
def work_h(b):
    global _fd
    if _fd is None: _fd = os.open(zp, os.O_RDONLY)
    r0, r1 = b * 64, min(len(I['h_cells']), (b + 1) * 64); miss = 0
    h = np.full((r1 - r0, 398, 398, 3), fill['histology'], M['histology']['dtype'])
    for i, r in enumerate(range(r0, r1)):
        for k in range(4):
            s = I['h_sz'][r, k]
            if not s: miss += 1; continue
            a = dec('histology', rd(I['h_off'][r, k], s), 199 * 199 * 3, M['histology']['dtype']).reshape(199, 199, 3)
            y, x = divmod(k, 2); h[i, y * 199:(y + 1) * 199, x * 199:(x + 1) * 199] = a
    m = np.load(f'{out}/histology.npy', mmap_mode='r+'); m[r0:r1] = h; m.flush(); del m
    return 10**6 + b, miss
if __name__ == '__main__':
    done = set(); lp = out + '/done_blocks.txt'
    if os.path.exists(lp): done = {int(x) for x in open(lp).read().split()}
    nb = (N + BLK - 1) // BLK; nh = (len(I['h_cells']) + 63) // 64
    jobs = [('c', b) for b in range(nb) if b not in done] ; jobs_h = [('h', b) for b in range(nh) if 10**6 + b not in done]
    print(f'{len(jobs)} cell blocks + {len(jobs_h)} hist blocks to do (of {nb}+{nh})', flush=True)
    t0 = time.time(); tot_miss = 0; k = 0
    with Pool(nproc) as p, open(lp, 'a') as lf:
        def go(fn, bl):
            global tot_miss, k
            for b, miss in p.imap_unordered(fn, bl, 1):
                lf.write(f'{b}\n'); lf.flush(); os.fsync(lf.fileno()); tot_miss += miss; k += 1
                if k % 50 == 0: print(f'{k} blocks, {time.time()-t0:.0f}s, missing chunks {tot_miss}', flush=True)
        go(work, [b for _, b in jobs]); go(work_h, [b for _, b in jobs_h])
    print('DONE blocks', k, 'missing chunks (filled)', tot_miss, f'{time.time()-t0:.0f}s')
