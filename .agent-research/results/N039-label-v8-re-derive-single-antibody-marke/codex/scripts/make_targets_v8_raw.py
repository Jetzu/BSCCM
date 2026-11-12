"""N039: recover the authors' clipped affine map and apply local background.

Labels only; no images/predictions. All fits use same-draw split-0 hist_row<0.
Fit XS=(total-background_lowess)/shading above BOTH released marker/autofluor
floors, then assess the floor-clipped six-channel affine prediction on ALL clean
split-0 cells. Floors identify censored/nonnegative-unmixing boundary regimes;
the autofluor value is never a predictor of the new raw label. If r>=.99, apply
exactly that map/floor to XL=(total-background)/shading.

Predeclared fallback if r<.99: spectral least squares using positive-gate minus
other-cell XL mean (clipped nonnegative, unit norm), plus same-draw unstained XL
mean and two centered PCs. All references use clean split-0 only. Project XL on
the basis pseudoinverse; subtract its split-0 0.5th percentile and floor
at a tiny positive split-0 scale for positive-q10 v3 compatibility. Fallback is not used on current data; no downstream QC chooses it.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

CHANNELS = ['426-446', '500-550', '550-570', '585-625', '627-673', '690-']
MARKERS = ['CD45', 'CD123', 'CD19', 'CD56', 'CD14', 'CD16', 'HLA-DR', 'CD3']


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def channels(sm, background):
    x = np.column_stack([
        (sm[f'Fluor_{c}_total_raw'] - sm[f'Fluor_{c}_{background}']) /
        sm[f'Fluor_{c}_shading'] for c in CHANNELS]).astype(float)
    if not np.isfinite(x).all():
        raise ValueError('Non-finite channels or invalid shading')
    return x


def derive(source, out):
    source, out = Path(source), Path(out)
    if source.resolve() == out.resolve():
        raise ValueError('Output must be separate from read-only source')
    out.mkdir(parents=True, exist_ok=True)
    sm = pd.read_parquet(source / 'bsccm_sm.parquet')
    idx = pd.read_parquet(source / 'bsccm_index.parquet')
    with np.load(source / 'targets.npz') as t:
        cal = (t['split'] == 0) & (t['hist_row'] < 0)
    assert len(idx) == len(sm) == len(cal)
    assert np.array_equal(idx.global_index.to_numpy(), np.arange(len(idx)))
    xs, xl = channels(sm, 'background_lowess'), channels(sm, 'background')
    zs = np.column_stack([xs, np.ones(len(xs))])
    zl = np.column_stack([xl, np.ones(len(xl))])
    af = sm.autofluor_single_antibody_model_unmixed.to_numpy()
    ab, batch = idx.antibodies.to_numpy(), idx.batch.to_numpy()
    result = pd.DataFrame(index=sm.index)
    records, models = [], {}
    for m in MARKERS:
        col = f'{m}_single_antibody_model_unmixed'
        y = sm[col].to_numpy()
        value = y.copy()  # v3 never consumes unmeasured rows
        for b in (0, 1):
            slide = (ab == m) & (batch == b)
            tr = slide & cal
            floor, af_floor = float(y[tr].min()), float(af[tr].min())
            interior = tr & (y > floor * (1 + 1e-6)) & (af > af_floor * (1 + 1e-6))
            if interior.sum() < 50:
                raise ValueError(f'Too few interior cells: {m}, draw {b}')
            coef = np.linalg.lstsq(zs[interior], y[interior], rcond=None)[0]
            pred = np.maximum(floor, zs[tr] @ coef)
            r = corr(y[tr], pred)
            rec = dict(target=m, draw=b, n_cal=int(tr.sum()), n_fit=int(interior.sum()),
                       floor=floor, r=r, residual_sd=float(np.std(y[tr] - pred)),
                       target_sd=float(np.std(y[tr])),
                       floor_fraction=float(np.mean(y[tr] == floor)))
            # Also report the requested plain linear tests with/without autofluor.
            for key, z in [('ols6', zs), ('ols6_af', np.column_stack([xs, af, np.ones(len(xs))]))]:
                c = np.linalg.lstsq(z[tr], y[tr], rcond=None)[0]
                p = z[tr] @ c
                rec[key + '_r'] = corr(y[tr], p)
                rec[key + '_residual_sd'] = float(np.std(y[tr] - p))
            model = dict(coef=coef.tolist(), floor=floor, method='recovered_clipped_affine')
            if r >= .99:
                value[slide] = np.maximum(floor, zl[slide] @ coef)
            else:
                g = sm[f'selection_example_{m}_positive_cells_batch_{b}'].to_numpy().astype(bool)
                unst = (ab == 'unstained') & (batch == b) & cal
                if min((tr & g).sum(), (tr & ~g).sum(), unst.sum()) < 3:
                    raise ValueError('Too few spectral-reference cells')
                dye = np.maximum(xl[tr & g].mean(0) - xl[tr & ~g].mean(0), 0)
                dye /= np.linalg.norm(dye)
                mu = xl[unst].mean(0)
                pcs = np.linalg.svd(xl[unst] - mu, full_matrices=False)[2][:2]
                basis = np.column_stack([dye, mu / np.linalg.norm(mu), pcs.T])
                a = xl[slide] @ np.linalg.pinv(basis)[0]
                atr = a[cal[slide]]
                offset = float(np.percentile(atr, .5))
                local_floor = float(max(np.median(np.abs(atr)) * 1e-6, 1e-12))
                value[slide] = np.maximum(local_floor, a - offset)
                model = dict(method='spectral_ls_fallback', basis=basis.tolist(),
                             offset=offset, floor=local_floor)
            rec['method'] = model['method']
            records.append(rec)
            models[f'{m}|b{b}'] = model
            print(f'{m} draw {b}: r={r:.9f}, residual_sd={rec["residual_sd"]:.6g}, {rec["method"]}', flush=True)
        result[col] = value
    result.to_parquet(out / 'raw_v8.parquet', index=False)
    pd.DataFrame(records).to_csv(out / 'unmixing_fidelity.csv', index=False)
    (out / 'unmixing_maps.json').write_text(json.dumps(dict(
        channels=CHANNELS, fit='same-draw split-0 hist_row<0; exclude both floors',
        fidelity='all same-draw clean split-0 single-antibody cells', models=models), indent=2) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', default='/hnvme/workspace/v181bb10-bsccm')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    derive(args.source, args.out)
