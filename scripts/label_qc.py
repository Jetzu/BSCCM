"""Label QC per (slide, target) -- N023. Checks whether a fluorescence-derived label measures its marker, using cell classes that do not
read the label under test.
 1. LF-class pattern: lym/gran/mono assigned from label-free IMAGE features (logistic on frozen LF features, fit on batch-0 'all' gated cells,
    split 0; confident cells p>0.8). Provenance: the default features come from a teacher trained on v4 fluorescence labels (incl. the
    targets under test); --feats .../n007_ssl_aug/probe_fov/lf_feats_all.npy is the SSL-only control (no fluorescence supervision) and
    gives the same signs (red-team F3). Gates also come from the unmixing system; batch-1 classes have no independent annotation. Per slide: class means of the label and Cohen's d of a pre-registered canonical contrast
    (CANON below, fixed before looking at per-slide numbers). A label whose canonical contrast vanishes or flips does not measure its marker.
 2. Dye amplitude (single-antibody slides): raw 6-channel shading-corrected mean of the dataset's spectrum-estimation gate
    (selection_example_<m>_positive_cells_batch_<b>) minus the other cells of the slide, clipped at 0, divided by the IQR of same-batch
    unstained cells projected on that direction (SNR). Population-dependent estimate, not calibrated brightness (red-team F2).
 3. Model r per batch from a FOV-split teacher's eval predictions (diagnostic only; NOT used for the QC decision).
Writes <out>/label_qc.csv and <out>/label_qc_class_means.json."""
import argparse, json, numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from bsccm.data import W
ap = argparse.ArgumentParser(); ap.add_argument('--feats', default=W + '/runs/n022_base_fov/probe/lf_feats_all.npy')
ap.add_argument('--preds', default=W + '/runs/n022_tail05_fov/eval_preds.npz'); ap.add_argument('--targets', default='targets_v4.npz')
ap.add_argument('--per-slide', type=int, default=8000); ap.add_argument('--out', required=True); a = ap.parse_args()
CANON = {'single:CD45': ('lym', 'gran'), 'single:CD19': ('lym', 'gran'), 'single:CD56': ('lym', 'gran'), 'single:CD14': ('mono', 'lym'),
         'single:CD16': ('gran', 'lym'), 'single:HLA-DR': ('mono', 'gran'), 'single:CD3': ('lym', 'gran'), 'single:CD123': None,
         'grp:CD123/HLA-DR/CD14': ('mono', 'lym'), 'grp:CD3/CD19/CD56': ('lym', 'mono'), 'grp:CD45': ('lym', 'gran'), 'grp:CD16': ('gran', 'lym'),
         'autofl': None}
CL = ['lym', 'gran', 'mono']
T = np.load(W + '/' + a.targets, allow_pickle=True); Y, M, sl, sn, b, c3, sp = T['Y'], T['mask'], T['slide'], T['slide_names'].astype(str), T['batch'], T['cls3'], T['split']
names = [str(n) for n in T['names']]; rng = np.random.default_rng(0)
rows = np.sort(np.concatenate([rng.permutation(np.where(sl == s)[0])[:a.per_slide] for s in np.unique(sl)]))
F = np.load(a.feats, mmap_mode='r'); X = np.concatenate([np.asarray(F[rows[i:i + 20000]], dtype=np.float32) for i in range(0, len(rows), 20000)])
pca = PCA(128, random_state=0).fit(X[::4]); Z = pca.transform(X)
trn = (b[rows] == 0) & (c3[rows] >= 0) & (sp[rows] == 0); val = (b[rows] == 0) & (c3[rows] >= 0) & (sp[rows] != 0)
clf = LogisticRegression(max_iter=3000).fit(Z[trn], c3[rows][trn]); acc = float((clf.predict(Z[val]) == c3[rows][val]).mean())
P = clf.predict_proba(Z); cls = np.where(P.max(1) > 0.8, P.argmax(1), -1)
print(f'LF cls3 classifier: held-out-FOV acc (batch 0) {acc:.3f}; confident {np.mean(cls >= 0):.2f}')
# dye amplitude on single-antibody slides
sm = pd.read_parquet(W + '/bsccm_sm.parquet'); ix = pd.read_parquet(W + '/bsccm_index.parquet')
raw = [c for c in sm.columns if c.endswith('_shading_corrected')]; R = sm[raw].values.astype(np.float64); ab = ix['antibodies'].values.astype(str)
def snr(slide_rows, marker, bb):
    g = sm[f'selection_example_{marker}_positive_cells_batch_{bb}'].values.astype(bool)[slide_rows]
    if g.sum() < 20 or (~g).sum() < 20: return np.nan, np.nan
    d = np.clip(np.nanmean(R[slide_rows][g], 0) - np.nanmean(R[slide_rows][~g], 0), 0, None); e = d / (np.linalg.norm(d) + 1e-12)
    un = np.where((ab == 'unstained') & (ix['batch'].values == bb))[0]; pr = R[un] @ e
    spread = np.nanpercentile(pr, 75) - np.nanpercentile(pr, 25)
    return float(np.linalg.norm(d)), float(np.linalg.norm(d) / (spread + 1e-9))
# model r per batch (diagnostic)
E = np.load(a.preds); Pm, Gm = E['P'], E['G']
out, cm = [], {}
for j, n in enumerate(names):
    for s in np.unique(sl):
        r_all = np.where((sl == s) & M[:, j])[0]
        if len(r_all) < 500: continue
        rr = rows[(sl[rows] == s) & M[rows, j]]; k = cls[np.searchsorted(rows, rr)]; y = Y[rr, j]
        mu = {CL[c]: float(y[k == c].mean()) if (k == c).sum() > 30 else np.nan for c in range(3)}
        ok = k >= 0; cmean = np.array([mu[CL[c]] for c in k[ok]]); eta = float(np.corrcoef(y[ok], cmean)[0, 1]) if ok.sum() > 100 else np.nan
        d = np.nan
        if CANON[n]:
            hi_, lo_ = CL.index(CANON[n][0]), CL.index(CANON[n][1]); y1, y0 = y[k == hi_], y[k == lo_]
            if len(y1) > 30 and len(y0) > 30: d = float((y1.mean() - y0.mean()) / np.sqrt((y1.var() + y0.var()) / 2))
        amp, sn_ = (np.nan, np.nan)
        if n.startswith('single:'): amp, sn_ = snr(np.where(sl == s)[0], n.split(':')[1], int(b[r_all[0]]))
        mt = np.isin(Gm, r_all); mr = float(np.corrcoef(Pm[mt, j], Y[Gm[mt], j])[0, 1]) if mt.sum() > 200 else np.nan
        out.append(dict(target=n, slide=sn[s].replace('antibodies_', ''), batch=int(b[r_all[0]]), n=len(r_all), canon=('>'.join(CANON[n]) if CANON[n] else ''),
                        canon_d=d, class_eta=eta, dye_amp=amp, dye_snr=sn_, model_r_fov=mr, **{f'mean_{c}': mu[c] for c in CL}))
df = pd.DataFrame(out); df.to_csv(a.out + '/label_qc.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(df.drop(columns=['n']).round(2).to_string(index=False))
json.dump(dict(cls3_acc_batch0=acc, canon=CANON, feats=a.feats, preds=a.preds), open(a.out + '/label_qc_meta.json', 'w'), indent=1)
