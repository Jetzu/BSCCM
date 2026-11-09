"""N023: per-batch dye signal estimates + crosstalk controls (red-team F2 fix).
Spectra: raw 6-channel shading-corrected means of the dataset's own spectrum-estimation gates (selection_example_<m>_positive_cells_batch_<b>)
minus the other cells of the same single-antibody slide, clipped at 0 (dye adds light; negative components are population composition).
b1/b0 norm ratios are population-dependent estimates, NOT calibrated brightness.
Crosstalk control: the released pooled CD123/HLA-DR/CD14 full-model unmixed value on slides WITHOUT that antibody group (CD16-only,
CD45-only, unstained) vs on 'all' slides, per batch."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ix = pd.read_parquet(W + '/bsccm_index.parquet')
raw = [c for c in sm.columns if c.endswith('_shading_corrected')]; R = sm[raw].values.astype(np.float64)
ab = ix['antibodies'].values.astype(str); b = ix['batch'].values
np.set_printoptions(precision=1, suppress=True, linewidth=200); print('channels', [c.split('_')[1] for c in raw]); sig = {}
for m in ['CD123', 'HLA-DR', 'CD14', 'CD3', 'CD19', 'CD56', 'CD45', 'CD16']:
    for bb in (0, 1):
        s = (ab == m) & (b == bb); g = sm[f'selection_example_{m}_positive_cells_batch_{bb}'].values.astype(bool)
        if (s & g).sum() < 20 or (s & ~g).sum() < 20: print(f'{m:7s} b{bb}: too few gated cells'); continue
        d = np.clip(np.nanmean(R[s & g], 0) - np.nanmean(R[s & ~g], 0), 0, None); sig[(bb, m)] = d
        print(f'{m:7s} b{bb}: n_pos {(s & g).sum():5d}  pos-other (clipped) {d}  |d| {np.linalg.norm(d):.1f}')
    if (0, m) in sig and (1, m) in sig:
        x, y = sig[(0, m)], sig[(1, m)]; print(f'        cos b0/b1 {x @ y / np.linalg.norm(x) / np.linalg.norm(y):.3f}  norm ratio b1/b0 {np.linalg.norm(y) / np.linalg.norm(x):.2f}')
print('\ncrosstalk control: pooled CD123/HLA-DR/CD14 full-model unmixed value')
A = sm['CD123/HLA-DR/CD14_full_model_unmixed'].values
for bb in (0, 1):
    for a in ['all', 'CD16', 'CD45', 'CD14', 'unstained']:
        x = A[(ab == a) & (b == bb)]; x = x[np.isfinite(x)]
        print(f'  b{bb} {a:9s} n {len(x):6d}  q50 {np.percentile(x, 50):7.2f}  q90 {np.percentile(x, 90):7.2f}')
if len(sys.argv) > 1: np.save(sys.argv[1], {str(k): v for k, v in sig.items()}, allow_pickle=True)
