"""N047 AB (a3): is the FOV-level label offset visible in the LF images' BACKGROUND? For every split-2 FOV of the 8 single-antibody slides
(both draws) read up to 4 cells' LED crops (23 channels) and DPC: border = top+bottom 8 rows (cell-free surround). FOV feature = mean border
intensity per channel (raw ADU, int16 view as in GPUNorm) + DPC border mean/sd; z-scored within slide (label-free, test-time).
Then (1) weighted corr with the v5 residual FOV offset (10-model ensemble), (2) within-draw 2-fold CV R2 (diagnostic) and
(3) INDUCTIVE: ridge fitted on the other draw's FOVs, applied to this draw; pooled r change; also vs the authors' FOV-level autofl / 690- bg."""
import sys, numpy as np, pandas as pd, time
from sklearn.linear_model import RidgeCV
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); names = [str(n) for n in T['names']]
Y, M, bt, fov, sp, hr, sl = T['Y'], T['mask'], T['batch'], T['fov'], T['split'], T['hist_row'], T['slide']
ix = pd.read_parquet(W + '/bsccm_index.parquet'); ab = ix.antibodies.values.astype(str); sm = pd.read_parquet(W + '/bsccm_sm.parquet')
af = np.arcsinh(sm['autofluor_single_antibody_model_unmixed'].values / 50); bg5 = (sm['Fluor_690-_background'].values / sm['Fluor_690-_shading'].values)
led = np.load(W + '/pack/led_array.npy', mmap_mode='r'); dpc = np.load(W + '/pack/dpc.npy', mmap_mode='r')
fam = ['n031_base', 'n031_zoom2', 'n037_base', 'n037_zoom2', 'n040_student_base', 'n040_student_zoom2', 'n041_phys', 'n041_phystail', 'n046_student2_base', 'n046_student2_zoom2']
ENS = {}
for d, b in (('b10', 0), ('b01', 1)):
    Ps = [np.load(f'{W}/runs/{m}_{d}/eval_preds.npz') for m in fam]; ENS[b] = (np.mean([p['P'] for p in Ps], 0), Ps[0]['G'])
MK = ['CD19', 'CD56', 'CD3', 'CD123', 'HLA-DR', 'CD14', 'CD45', 'CD16']
sel = np.flatnonzero(np.isin(ab, MK) & (sp == 2) & (hr < 0))
rng = np.random.default_rng(0); df = pd.DataFrame(dict(i=sel, f=fov[sel])); df['r'] = rng.random(len(df))
pick = np.sort(df.sort_values("r").groupby("f").head(4).i.values.copy())
t0 = time.time(); feats = np.zeros((len(pick), 25), np.float32)
for n, i in enumerate(pick):
    x = np.asarray(led[i]).view(np.int16).astype(np.float32); bd = np.concatenate([x[:, :8, :], x[:, -8:, :]], 1)
    feats[n, :23] = bd.reshape(23, -1).mean(1)
    dd = np.asarray(dpc[i]); db = np.concatenate([dd[:8], dd[-8:]]); feats[n, 23] = db.mean(); feats[n, 24] = db.std()
print(f'read {len(pick)} crops in {time.time()-t0:.0f}s', flush=True)
LEDN = ["BF","DF_50","DF_50_B","DF_50_R","DF_55","DF_60","DF_60_B","DF_60_R","DF_65","DF_70","DF_70_B","DF_70_R","DF_75","DF_80","DF_80_B","DF_80_R","DF_85","DF_90","DPC_B","DPC_L","DPC_R","DPC_T","LED119","dpc_mean","dpc_sd"]
Fc = pd.DataFrame(np.log(np.clip(feats[:, :23], 1, None)), columns=LEDN[:23]); Fc['dpc_mean'] = feats[:, 23]; Fc['dpc_sd'] = feats[:, 24]; Fc['f'] = fov[pick]
FF = Fc.groupby('f').mean(); FF['slide'] = pd.Series(sl[pick], index=fov[pick]).groupby(level=0).first()
FX = FF[LEDN].copy()
for s in FF.slide.unique():
    m = (FF.slide == s).values; FX.loc[m] = (FX.loc[m] - FX.loc[m].mean()) / (FX.loc[m].std() + 1e-9)
FX = FX.fillna(0)
def offs(m, b):
    j = names.index(f'single:{m}'); P, G = ENS[b]; pos = pd.Series(np.arange(len(G)), index=G)
    s = np.flatnonzero((ab == m) & (bt == b) & M[:, j] & (sp == 2) & (hr < 0)); s = s[np.isin(s, G)]; p = P[pos[s].values, j]; y = Y[s, j]
    yz = (y - y.mean()) / y.std(); pz = (p - p.mean()) / p.std(); bb, aa = np.polyfit(pz, yz, 1); e = yz - aa - bb * pz
    g = pd.DataFrame(dict(f=fov[s], e=e, af=af[s], bg5=bg5[s])).groupby('f').agg(o=('e', 'mean'), n=('e', 'size'), af=('af', 'mean'), bg5=('bg5', 'mean'))
    g = g[g.index.isin(FX.index)]; return dict(s=s, f=fov[s], y=yz, p=pz, b=bb, g=g)
rows = []
for m in MK:
    R = {b: offs(m, b) for b in (0, 1)}
    for b in (0, 1):
        g = R[b]['g']; X = FX.loc[g.index].values; o = g.o.values; w = g.n.values; go = R[1 - b]['g']; Xo = FX.loc[go.index].values
        r = dict(target=m, draw=b, n_fov=len(g))
        cc = {c: np.corrcoef(FX.loc[g.index, c].values, o)[0, 1] for c in LEDN}; best = max(cc, key=lambda c: abs(cc[c])); r['best_led_corr'] = f'{best}:{cc[best]:.2f}'
        r['max_abs_led_corr'] = abs(cc[best]); r['corr_af_off'] = np.corrcoef(g.af, o)[0, 1]
        ccaf = {c: np.corrcoef(FX.loc[g.index, c].values, g.af.values)[0, 1] for c in LEDN}; ba = max(ccaf, key=lambda c: abs(ccaf[c])); r['best_led_vs_authors_autofl'] = f'{ba}:{ccaf[ba]:.2f}'
        ccb = {c: np.corrcoef(FX.loc[g.index, c].values, g.bg5.values)[0, 1] for c in LEDN}; bb5 = max(ccb, key=lambda c: abs(ccb[c])); r['best_led_vs_bg690'] = f'{bb5}:{ccb[bb5]:.2f}'
        s0 = np.average((o - np.average(o, weights=w)) ** 2, weights=w)
        fold = rng.integers(0, 2, len(o)); pr = np.zeros(len(o))
        for k in (0, 1):
            pr[fold == k] = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(X[fold != k], o[fold != k], sample_weight=w[fold != k]).predict(X[fold == k])
        r['cvR2'] = 1 - np.average((o - pr) ** 2, weights=w) / s0
        pi = RidgeCV(alphas=np.logspace(-1, 5, 13)).fit(Xo, go.o.values, sample_weight=go.n.values).predict(X); r['indR2'] = 1 - np.average((o - pi) ** 2, weights=w) / s0
        # pooled r with inductive correction (cells in FOVs with features)
        S = R[b]; mp = pd.Series(pi, index=g.index); ok = np.isin(S['f'], g.index); corr = np.zeros(len(S['f'])); corr[ok] = mp[S['f'][ok]].values
        r['r_pool'] = np.corrcoef(S['p'], S['y'])[0, 1]; r['r_pool_ind_corrected'] = np.corrcoef(S['p'] + corr / S['b'], S['y'])[0, 1]
        rows.append(r)
out = pd.DataFrame(rows); out.to_csv(sys.argv[1] + '/a3_led_border.csv', index=False); pd.set_option('display.width', 250)
print(out.round(3).to_string(index=False))
print('mean delta pooled r (inductive LED-border correction), 16 single target x draw: %.4f' % (out.r_pool_ind_corrected - out.r_pool).mean())
