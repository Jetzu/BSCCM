"""N034 fluorescence IMAGE targets for LF -> marker-image generation (flow matching).
Per cell x channel (6 fluor channels, 128 px): background = median of the 6-px border ring, integrals of (x - bg) in a disk r<=56 and in
annuli r<16 / 16-32 / 32-56, border noise (1.4826 * MAD). Then a slide-POSITION gain (K007, user ruling: position carries no
biology) is removed multiplicatively from the IMAGE targets where it is real: smooth Huber model per slide x channel (see __main__),
fitted on split-0 clean cells, kept only if it generalises to held-out split-1 FOVs (red-team F2); gain clipped to [1/3, 3]. Targets are
(x - bg) / gain; inputs (LF) are never touched. Writes W/fm_prep.npz:
  bg, gain, noise (N,6), integ (N,6,4: disk + 3 annuli, bg-subtracted, NOT gain-corrected) float32; a_c (6,) per-channel pixel noise
  scale (median border noise over split-0 clean cells of --a-c-batch); stain (N,) int8 index into stain_names; gain_info json.
CPU job: 48 workers for the 81 GB read, OMP capped before imports (N029 lesson)."""
import os; os.environ.setdefault('OMP_NUM_THREADS', '2')
import json, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool
W = '/hnvme/workspace/v181bb10-bsccm'
yy, xx = np.mgrid[:128, :128]; R = np.hypot(yy - 63.5, xx - 63.5)
BORDER = (np.minimum(np.minimum(yy, 127 - yy), np.minimum(xx, 127 - xx)) < 6)
MASKS = np.stack([R <= 56, R < 16, (R >= 16) & (R < 32), (R >= 32) & (R <= 56)]).astype(np.float32)   # (4,128,128)

def _chunk(args):
    a, b = args
    F = np.load(W + '/pack/fluor.npy', mmap_mode='r'); X = np.asarray(F[a:b], np.float32)        # (n,6,128,128)
    ring = X[:, :, BORDER]                                                                           # (n,6,P)
    bg = np.median(ring, -1); noise = 1.4826 * np.median(np.abs(ring - bg[..., None]), -1)
    integ = np.einsum('ncyx,kyx->nck', X - bg[..., None, None], MASKS)
    return a, bg.astype(np.float32), noise.astype(np.float32), integ.astype(np.float32)

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--recompute', action='store_true', help='re-read the 81 GB fluor pack (else reuse bg/noise/integ)')
    ap.add_argument('--a-c-batch', type=int, default=1, help='blood draw whose split-0 clean cells define a_c (= the training draw)')
    ap.add_argument('--accept', type=float, default=0.99, help='keep a slide x channel gain only if held-out (split 1) var ratio < this')
    a = ap.parse_args()
    t0 = time.time(); T = np.load(W + '/targets_v5c.npz', allow_pickle=True); N = len(T['split'])
    idx = pd.read_parquet(W + '/bsccm_index.parquet'); assert len(idx) == N and np.array_equal(idx.global_index.values, np.arange(N))
    if a.recompute or not os.path.exists(W + '/fm_prep.npz'):
        bg = np.zeros((N, 6), np.float32); noise = np.zeros((N, 6), np.float32); integ = np.zeros((N, 6, 4), np.float32)
        jobs = [(s, min(s + 512, N)) for s in range(0, N, 512)]
        with Pool(int(os.environ.get('SLURM_CPUS_PER_TASK', 16))) as pool:
            for k, (s, b_, n_, i_) in enumerate(pool.imap_unordered(_chunk, jobs, chunksize=1)):
                m = len(b_); bg[s:s + m], noise[s:s + m], integ[s:s + m] = b_, n_, i_
                if k % 100 == 0: print(f'{k}/{len(jobs)} {time.time() - t0:.0f}s', flush=True)
        print('read done', f'{time.time() - t0:.0f}s', flush=True)
    else:
        old = np.load(W + '/fm_prep.npz'); bg, noise, integ = old['bg'], old['noise'], old['integ']; print('reusing bg/noise/integ', flush=True)
    clean = T['hist_row'] < 0; sp = T['split']
    # N034 red-team F2: the first HGB gain (FOV-centre leaves) fitted FOV-composition noise (held-out var ratio .994 median, 25 % worse).
    # Now: smooth multiplicative model per slide x channel (quadratic in position-in-FOV + quadratic in stage x/y + linear z = optics /
    # staining gradients, cannot isolate FOVs), Huber fit on split-0 clean cells; kept ONLY if it reduces log-integral variance on held-out
    # split-1 FOVs by > 1 - accept; otherwise gain = 1. Split 2 (scored cells) is never used for fitting or acceptance.
    from sklearn.linear_model import HuberRegressor
    px = idx.position_in_fov_x_pix.values.astype(float); py = idx.position_in_fov_y_pix.values.astype(float)
    fx, fy, fz = (idx[c].values.astype(float) for c in ('fov_center_x', 'fov_center_y', 'fov_center_z'))
    slide = idx.data_path.values; gain = np.ones((N, 6), np.float32); info = {}
    def design(m):
        u = (px[m] - px.mean()) / px.std(); v = (py[m] - py.mean()) / py.std()
        X_, Y_ = fx[m], fy[m]; X_ = (X_ - X_.mean()) / (X_.std() + 1e-9); Y_ = (Y_ - Y_.mean()) / (Y_.std() + 1e-9); Z_ = (fz[m] - fz[m].mean()) / (fz[m].std() + 1e-9)
        return np.c_[u, v, u * u, v * v, u * v, X_, Y_, X_ * X_, Y_ * Y_, X_ * Y_, Z_]
    for s in sorted(set(slide)):
        m = np.where(slide == s)[0]; D = design(m)
        for c in range(6):
            I = integ[m, c, 0]; pos = I > 0; fit = (sp[m] == 0) & clean[m] & pos; hv = (sp[m] == 1) & clean[m] & pos; te = (sp[m] == 2) & clean[m] & pos
            key = f'{s}|{c}'
            if fit.sum() < 1000 or hv.sum() < 100: info[key] = dict(n_fit=int(fit.sum()), accepted=False, reason='too few'); continue
            h = HuberRegressor(max_iter=500, alpha=1e-3).fit(D[fit], np.log(I[fit])); tr = h.predict(D); tr -= np.median(tr[fit])
            lg = np.log(I); vr = lambda q: float(np.var(lg[q] - tr[q]) / np.var(lg[q]))
            e = dict(n_fit=int(fit.sum()), var_ratio_fit=vr(fit), var_ratio_heldout_split1=vr(hv), var_ratio_split2_report=vr(te) if te.sum() > 50 else None)
            e['accepted'] = bool(e['var_ratio_heldout_split1'] < a.accept)
            if e['accepted']:
                gain[m, c] = np.exp(np.clip(tr, -np.log(3), np.log(3))); e.update(gain_p5=float(np.percentile(gain[m, c], 5)), gain_p95=float(np.percentile(gain[m, c], 95)))
            info[key] = e
        print(s, {c: (round(info[f'{s}|{c}'].get('var_ratio_heldout_split1', np.nan), 3), info[f'{s}|{c}']['accepted']) for c in range(6)}, flush=True)
    stain_names = np.array(sorted(set(idx.antibodies.astype(str))))
    stain = np.searchsorted(stain_names, idx.antibodies.astype(str).values).astype(np.int8)
    a_c = np.median(noise[clean & (sp == 0) & (T['batch'] == a.a_c_batch)], 0).astype(np.float32)
    np.savez(W + '/fm_prep.npz.tmp.npz', bg=bg, gain=gain, noise=noise, integ=integ, a_c=a_c, stain=stain, stain_names=stain_names,
             gain_info=json.dumps(info), a_c_batch=a.a_c_batch)
    os.replace(W + '/fm_prep.npz.tmp.npz', W + '/fm_prep.npz')
    acc = [v.get('accepted', False) for v in info.values()]
    print('a_c', a_c, 'accepted gains', int(np.sum(acc)), '/', len(acc), 'stains', stain_names.tolist(), f'done {time.time() - t0:.0f}s')
