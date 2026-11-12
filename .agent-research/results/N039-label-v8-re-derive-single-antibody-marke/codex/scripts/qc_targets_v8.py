"""N039 frozen stage-1 label QC. See qc_protocol.md; never load predictions.

Only output reports are written; calibration files/data are read-only. The
primary d check is unassessable on the supplied single-antibody slides. A
fluorescence-only low-CD16/pooled-CD14 proxy is provided as a diagnostic, not
claimed as a verified lymphocyte population. GMM/proxy fits use split-0 only.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from make_targets_v8_raw import MARKERS, channels, corr


def fov_excess(y, f, seed):
    _, inv = np.unique(f, return_inverse=True)
    counts = np.bincount(inv)
    keep = counts[inv] >= 3
    y = np.asarray(y[keep], float)
    _, inv = np.unique(inv[keep], return_inverse=True)
    counts = np.bincount(inv)
    if len(y) < 3 or np.var(y) == 0:
        return dict(n_fov_cells=len(y), n_fovs=len(counts), share=np.nan,
                    null=np.nan, null_sd=np.nan, excess=np.nan)
    def share(v):
        return float(np.var((np.bincount(inv, weights=v) / counts)[inv]) / np.var(v))
    observed = share(y)
    rng = np.random.default_rng(seed)
    null = np.array([share(rng.permutation(y)) for _ in range(20)])
    return dict(n_fov_cells=len(y), n_fovs=len(counts), share=observed,
                null=float(null.mean()), null_sd=float(null.std()),
                excess=float((observed - null.mean()) / (1 - null.mean())))


def finite_corr(a, b):
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 3 or min(np.std(a[keep]), np.std(b[keep])) == 0:
        return np.nan
    return corr(a[keep], b[keep])


def evaluate(source, target, out, proxy_as_primary=False):
    source, target, out = Path(source), Path(target), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t = dict(np.load(target, allow_pickle=True))
    old = dict(np.load(source / 'targets_v5.npz', allow_pickle=True))
    xfit = dict(np.load(source / 'targets_v5cx.npz', allow_pickle=True))
    sm = pd.read_parquet(source / 'bsccm_sm.parquet')
    ix = pd.read_parquet(source / 'bsccm_index.parquet')
    xl, xs = channels(sm, 'background'), channels(sm, 'background_lowess')
    names = list(t['names'])
    assert names == list(old['names']) == list(xfit['names'])
    for key in ('batch', 'split', 'fov', 'hist_row'):
        assert np.array_equal(t[key], old[key]) and np.array_equal(t[key], xfit[key])
    y, mask, batch, sp, hr, fov = (t[k] for k in ('Y', 'mask', 'batch', 'split', 'hist_row', 'fov'))
    clean = hr < 0
    gates = sm.selection_lymphocytes.to_numpy().astype(bool)
    cd16 = sm.CD16_full_model_unmixed.to_numpy()
    cd14 = sm['CD123/HLA-DR/CD14_full_model_unmixed'].to_numpy()
    records = []
    for mi, m in enumerate(MARKERS):
        j = names.index('single:' + m)
        for b in (0, 1):
            sel = mask[:, j] & (batch == b) & clean & np.isfinite(y[:, j])
            assert np.all(ix.antibodies.to_numpy()[sel] == m)
            tr = sel & (sp == 0)
            te = sel & (sp == 2)
            r = dict(target=m, draw=b, n=int(sel.sum()), n_cal=int(tr.sum()), n_test=int(te.sum()))
            seed = 3900 + 2 * mi + b
            r.update({'fov_' + k: v for k, v in fov_excess(y[sel, j], fov[sel], seed).items()})
            r.update({'test_fov_' + k: v for k, v in fov_excess(y[te, j], fov[te], seed).items()})
            r['a_status'] = 'PASS' if r['fov_excess'] <= .05 else 'FAIL'
            r['frozen_v5_fov_excess'] = fov_excess(old['Y'][sel, j], fov[sel], seed)['excess']
            common = sel & old['mask'][:, j]
            r['r_v8_v5'] = finite_corr(y[common, j], old['Y'][common, j])
            r['r_v8_v5cx'] = finite_corr(y[sel, j], xfit['Y'][sel, j])
            r['test_r_v8_v5'] = finite_corr(y[te, j], old['Y'][te, j])
            r['c_status'] = ('PASS' if r['r_v8_v5'] >= .95 else 'FAIL') if m in ('CD16', 'CD45', 'CD14') else 'N/A'
            r['b_status'] = 'N/A'
            if m == 'CD19':
                gm = GaussianMixture(2, random_state=0, n_init=3, reg_covar=1e-6).fit(y[tr, j, None])
                hi = int(np.argmax(gm.means_[:, 0])); lo = 1 - hi
                means, var = gm.means_[:, 0], gm.covariances_.reshape(-1)
                r.update(cd19_positive_fraction=float(gm.weights_[hi]),
                         cd19_dprime=float((means[hi] - means[lo]) / np.sqrt(var[lo])),
                         cd19_dprime_pooled=float((means[hi] - means[lo]) / np.sqrt(var.mean())),
                         cd19_gmm_converged=bool(gm.converged_),
                         cd19_mu_negative=float(means[lo]), cd19_mu_positive=float(means[hi]),
                         cd19_var_negative=float(var[lo]), cd19_var_positive=float(var[hi]))
                r['b_status'] = 'PASS' if (gm.converged_ and .01 <= r['cd19_positive_fraction'] <= .06 and r['cd19_dprime'] >= 3) else 'FAIL'
            dc = 1 if m in ('CD19', 'CD56', 'CD3') else 0
            dye = np.arcsinh(xl[:, dc] / 10)
            r['r_own_dye_all'] = finite_corr(y[sel, j], dye[sel])
            r['d_status'] = 'N/A'
            if b == 1 and m in ('HLA-DR', 'CD19', 'CD14'):
                r['authors_lymphocyte_gate_n'] = int((sel & gates).sum())
                # Fixed median intersection: an LF-independent crosstalk proxy.
                q16, q14 = np.median(cd16[tr]), np.median(cd14[tr])
                proxy = sel & (cd16 <= q16) & (cd14 <= q14)
                r.update(proxy_threshold_cd16=float(q16), proxy_threshold_pooled_cd14=float(q14),
                         proxy_n=int(proxy.sum()), proxy_test_n=int((proxy & te).sum()),
                         proxy_r_own_dye=finite_corr(y[proxy, j], dye[proxy]),
                         proxy_r_own_dye_linear=finite_corr(y[proxy, j], xl[proxy, dc]),
                         proxy_r_own_dye_lowess=finite_corr(y[proxy, j], np.arcsinh(xs[proxy, dc] / 10)),
                         proxy_v5_r_own_dye=finite_corr(old['Y'][proxy, j], dye[proxy]),
                         proxy_test_r_own_dye=finite_corr(y[proxy & te, j], dye[proxy & te]))
                r['proxy_d_status'] = 'PASS' if r['proxy_r_own_dye'] >= 0 else 'FAIL'
                r['d_status'] = r['proxy_d_status'] if proxy_as_primary else 'FAIL'
                r['d_reason'] = ('user-authorized fluorescence-only proxy' if proxy_as_primary else
                                 'unassessable: no gated single-antibody lymphocytes; no same-cell CD16/CD14')
            r['overall_status'] = 'FAIL' if 'FAIL' in [r[k + '_status'] for k in 'abcd'] else 'PASS'
            records.append(r)
            print(m, b, {k: r[k] for k in ('fov_excess', 'a_status', 'b_status', 'r_v8_v5', 'c_status', 'd_status')}, flush=True)
    df = pd.DataFrame(records)
    df.to_csv(out / 'qc_v8.csv', index=False)
    all_pass = bool((df.overall_status == 'PASS').all())
    metadata = dict(overall='PASS' if all_pass else 'FAIL', source=str(source), candidate=str(target),
                    baseline='targets_v5.npz (frozen)', sensitivity_baseline='targets_v5cx.npz',
                    cohort='all measured finite hist_row<0 rows, splits 0/1/2',
                    gmm_fit='clean split-0 only; same draw', proxy_as_primary=proxy_as_primary,
                    no_models_scored=True, criterion_counts={k: df[k + '_status'].value_counts().to_dict() for k in 'abcd'})
    (out / 'qc_v8_meta.json').write_text(json.dumps(metadata, indent=2) + '\n')
    lines = ['# N039 pre-registered v8 QC', '', '**Overall: ' + metadata['overall'] + ' — stop; v8 is not approved for stage-2 training/scoring.**' if not all_pass else '**Overall: PASS.**', '',
             'Candidate: local-background raw → clean v3 → v4 FOV xfit → v5 mask/normalization (`targets_v8cx.npz`). '
             'No model predictions, LF images, or LF features were loaded. Thresholds were not tuned.', '',
             'Primary cohort is all clean measured single-antibody cells across splits. Map/calibration, GMM, and proxy thresholds fit clean split-0 of the same draw. '
             'See `qc_protocol.md` for formulas and fixed seeds; CSV includes split-2 sensitivities.', '',
             '| Target | Draw | n | (a) FOV excess | (a) | (b) CD19 fraction / d′ | (b) | r(v8,v5) | (c) | (d) |',
             '|---|---:|---:|---:|---|---:|---|---:|---|---|']
    for r in records:
        mix = f"{100*r['cd19_positive_fraction']:.2f}% / {r['cd19_dprime']:.3f}" if r['target'] == 'CD19' else '—'
        lines.append(f"| {r['target']} | {r['draw']} | {r['n']} | {r['fov_excess']:.4f} | {r['a_status']} | {mix} | {r['b_status']} | {r['r_v8_v5']:.4f} | {r['c_status']} | {r['d_status']} |")
    lines += ['', '(a) excess ≤0.05 (20-permutation null; FOVs ≥3 cells). (b) CD19 positive weight 1–6% AND d′≥3 in both draws, '
              'd′=(μpos−μneg)/σneg. (c) r≥0.95 for CD16/CD45/CD14 against frozen v5; other correlations are descriptive.', '',
              '(d) Authors’ lymphocyte gates cover zero cells on these single-antibody slides. True same-cell CD16/CD14 labels are absent. '
              + ('The user authorized the fluorescence-only low-CD16/pooled-CD14 proxy as primary; its biological specificity is unverified.' if proxy_as_primary else
                 '**Required check FAIL/unassessable**; a fluorescence-only proxy below is diagnostic and does not establish a lymphocyte-specific PASS.'), '',
              'The proxy intersects values ≤same-slide clean split-0 medians for released `CD16_full_model_unmixed` and '
              '`CD123/HLA-DR/CD14_full_model_unmixed`. These can be crosstalk on slides lacking those antibodies; the latter is a pooled group. '
              'Own dye is asinh(XL/10): 500-550 (CD19), 426-446 (HLA-DR/CD14).', '',
              '| Draw-1 target | Proxy n | r(v8, own dye) | Sign test | r(v5, own dye) | Primary (d) |',
              '|---|---:|---:|---|---:|---|']
    for r in records:
        if 'proxy_n' in r:
            lines.append(f"| {r['target']} | {r['proxy_n']} | {r['proxy_r_own_dye']:.4f} | {r['proxy_d_status']} | {r['proxy_v5_r_own_dye']:.4f} | {r['d_status']} |")
    lines += ['', 'Mixture weights are descriptive two-Gaussian fits, not externally validated B-cell frequencies. '
              'FOV excess measures clustering of the label; it cannot distinguish background error from actual FOV cell-composition differences. '
              'No masking, retuning, alternative candidate selection, or model scoring follows a failure.', '']
    (out / 'qc_v8.md').write_text('\n'.join(lines))
    return all_pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='/hnvme/workspace/v181bb10-bsccm')
    ap.add_argument('--targets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--proxy-as-primary', action='store_true', help='Only with explicit user authorization')
    args = ap.parse_args()
    evaluate(args.source, args.targets, args.out, args.proxy_as_primary)
