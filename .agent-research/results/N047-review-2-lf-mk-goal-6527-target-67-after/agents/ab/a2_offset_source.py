"""N047 AB (a2): what IS the FOV-level offset of the v5 CD19/CD56/CD3 labels? FOV-level decomposition on the TEST FOVs (split 2) of each
single-antibody slide: FOV mean of v5 label vs FOV means of v3 label, the v4 detrend trend (v3 - v4), dye-channel raw signal with the
authors' smoothed bg (XS) and per-cell bg (XL), bg error in all 6 channels, authors' unmixed value, authors' autofluor unmixed, z focus.
Also the FOV excess (over a cell-permutation null) of each quantity."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y5, Y3, Y4, M, bt, fov, sp, hr = T['Y'], T['Y_v3'], T['Y_v4'], T['mask'], T['batch'], T['fov'], T['split'], T['hist_row']
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ix = pd.read_parquet(W + '/bsccm_index.parquet'); ab = ix.antibodies.values.astype(str)
CH = ['426-446', '500-550', '550-570', '585-625', '627-673', '690-']
TOT = np.stack([sm[f'Fluor_{c}_total_raw'].values for c in CH], 1); BG = np.stack([sm[f'Fluor_{c}_background'].values for c in CH], 1)
BGL = np.stack([sm[f'Fluor_{c}_background_lowess'].values for c in CH], 1); SH = np.stack([sm[f'Fluor_{c}_shading'].values for c in CH], 1)
XL = (TOT - BG) / SH; XS = (TOT - BGL) / SH; BERR = (BG - BGL) / SH
rng = np.random.default_rng(0)
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
ENS = {}
for d, b in (('b10', 0), ('b01', 1)):
    Ps = [np.load(f'{W}/runs/{m}_{d}/eval_preds.npz') for m in fam]; ENS[b] = (np.mean([p['P'] for p in Ps], 0), Ps[0]['G'])
def fovmeans(x, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); return np.bincount(inv, x) / c, c, inv
def excess(x, f):
    u, inv = np.unique(f, return_inverse=True); c = np.bincount(inv); ok = c[inv] >= 2; x = x[ok]; inv = np.unique(inv[ok], return_inverse=True)[1]; c = np.bincount(inv)
    sh = np.var((np.bincount(inv, x) / c)[inv]) / np.var(x); nul = np.mean([np.var((np.bincount(inv, xx) / c)[inv]) / np.var(xx) for xx in (rng.permutation(x) for _ in range(10))])
    return (sh - nul) / (1 - nul)
out = []
for m in ['CD19', 'CD56', 'CD3', 'CD123', 'HLA-DR', 'CD14', 'CD45', 'CD16']:
    j = names.index(f'single:{m}'); dc = 1 if m in ('CD19', 'CD56', 'CD3') else 0
    un = sm[f'{m}_single_antibody_model_unmixed'].values; af = sm['autofluor_single_antibody_model_unmixed'].values
    for b in (0, 1):
        P, G = ENS[b]; pos = pd.Series(np.arange(len(G)), index=G)
        s = np.flatnonzero((ab == m) & (bt == b) & M[:, j] & (sp == 2) & (hr < 0)); s = s[np.isin(s, G)]
        p = P[pos[s].values, j]; f = fov[s]
        q = dict(v5=Y5[s, j], v3=Y3[s, j], trend=Y3[s, j] - Y4[s, j], pred=p, XS_dye=np.arcsinh(XS[s, dc] / 10), XL_dye=np.arcsinh(XL[s, dc] / 10),
                 authors_unmixed=np.arcsinh(un[s] / 50), autofl=np.arcsinh(af[s] / 50), z_focus=ix.fov_center_z.values[s].astype(float))
        for c in range(6): q[f'bgerr{c}'] = BERR[s, c]
        for c in range(6): q[f'bgl{c}'] = BGL[s, c] / SH[s, c]
        for c in range(6): q[f'bgloc{c}'] = BG[s, c] / SH[s, c]
        fm = {k: fovmeans(v, f)[0] for k, v in q.items()}; _, cn, inv = fovmeans(q['v5'], f); w = cn
        # residual offset of v5 after the prediction (as in a_fov_offsets): FOV mean of (z(y) - b z(p))
        yz = (q['v5'] - q['v5'].mean()) / q['v5'].std(); pz = (p - p.mean()) / p.std(); bb, aa = np.polyfit(pz, yz, 1); e = yz - aa - bb * pz; off = fovmeans(e, f)[0]
        keep = cn >= 3
        r = dict(target=m, draw=b, n_cells=len(s), n_fov=int(keep.sum()))
        for k in ['v5', 'v3', 'trend', 'XS_dye', 'XL_dye', 'authors_unmixed', 'autofl', 'bgerr0', 'bgerr1', 'pred']: r[f'exc_{k}'] = float(excess(q[k], f))
        def wc(a, b_): a, b_ = a[keep], b_[keep]; ww = w[keep]; a = a - np.average(a, weights=ww); b_ = b_ - np.average(b_, weights=ww); return float(np.average(a * b_, weights=ww) / np.sqrt(np.average(a * a, weights=ww) * np.average(b_ * b_, weights=ww)))
        for k in fm:
            if k != 'v5': r[f'cOFF_{k}'] = wc(off, fm[k])
        out.append(r)
df = pd.DataFrame(out); df.to_csv(sys.argv[1] + '/a2_offset_source.csv', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 80)
print(df[['target', 'draw', 'n_cells', 'n_fov'] + [c for c in df.columns if c.startswith('exc_')]].round(2).to_string(index=False))
print(df[['target', 'draw'] + [c for c in df.columns if c.startswith('cOFF_')]].round(2).to_string(index=False))
