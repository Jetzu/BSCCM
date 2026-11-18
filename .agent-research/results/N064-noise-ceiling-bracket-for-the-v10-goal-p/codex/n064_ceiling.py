#!/usr/bin/env python3
"""N064: CPU-only conditional pixel-noise reliability audit; source data are read-only."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import signal
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

W = Path('/hnvme/workspace/v181bb10-bsccm')
REPO = Path('/home/hpc/v181bb/v181bb10/BSCCM')
N039 = REPO / '.agent-research/results/N039-label-v8-re-derive-single-antibody-marke/codex/unmixing_maps.json'
CHANNELS = ['426-446', '500-550', '550-570', '585-625', '627-673', '690-']
MARKERS = ['CD45', 'CD123', 'CD19', 'CD56', 'CD14', 'CD16', 'HLA-DR', 'CD3']
GROUPS = ['CD123/HLA-DR/CD14', 'CD3/CD19/CD56', 'CD45', 'CD16']
COF = {m: (150 if m in ('CD45', 'CD16') else 500) for m in MARKERS}
STOP = False
FL = None
YY, XX = np.indices((128, 128))
FG = (YY - 64)**2 + (XX - 64)**2 < 64**2
PAR = (YY + XX) % 2
PIX = [(FG & (PAR == h), ~FG & (PAR == h)) for h in (0, 1)]
assert not np.any(PIX[0][0] & PIX[1][0]) and not np.any(PIX[0][1] & PIX[1][1])
assert all(not np.any(f & b) for f, b in PIX)


def corr(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return float('nan')
    return float(pearsonr(a[ok], b[ok]).statistic)


def atomic_json(path, obj):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def json_clean(obj):
    if isinstance(obj, np.ndarray):
        return json_clean(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): json_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_clean(v) for v in obj]
    if isinstance(obj, np.generic):
        return json_clean(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def stop_handler(signum, frame):
    global STOP
    STOP = True
    print(f'Signal {signum}: checkpointing completed pixel chunks, then exit 143', flush=True)


def init_worker(pack):
    global FL
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    FL = np.load(pack, mmap_mode='r')
    assert FL.ndim == 4 and FL.shape[1:] == (6, 128, 128)


def pixel_chunk(item):
    key, rows = item
    means = np.empty((len(rows), 2, 2, 6), dtype=np.float64)
    quality = np.zeros((len(rows), 2), dtype=np.int64)
    for k, g in enumerate(rows):
        image = np.asarray(FL[int(g)])
        quality[k] = [np.count_nonzero(image == 0), np.count_nonzero(image >= 4095)]
        for h, (fg, bg) in enumerate(PIX):
            means[k, h, 0] = image[:, fg].mean(axis=1, dtype=np.float64)
            means[k, h, 1] = image[:, bg].mean(axis=1, dtype=np.float64)
    return key, rows, means, quality


def pixel_cache(rows, out, pack, workers, chunk_size):
    cache = out / 'pixel_chunks'
    cache.mkdir(exist_ok=True)
    chunks = [(k, rows[a:a + chunk_size]) for k, a in enumerate(range(0, len(rows), chunk_size))]
    missing = []
    for key, rr in chunks:
        p = cache / f'{key:06d}.npz'
        if not p.exists():
            missing.append((key, rr))
        else:
            with np.load(p) as z:
                if not np.array_equal(z['G'], rr):
                    raise ValueError(f'Cache row mismatch: {p}')
    print(f'Pixel rows={len(rows)}, missing chunks={len(missing)}/{len(chunks)}, workers={workers}', flush=True)
    if missing:
        # fork is explicit: Linux container and uv smoke share the same path; no training dependencies.
        with mp.get_context('fork').Pool(workers, initializer=init_worker, initargs=(str(pack),)) as pool:
            for key, rr, means, quality in pool.imap_unordered(pixel_chunk, missing, chunksize=1):
                p = cache / f'{key:06d}.npz'
                tmp = p.with_suffix('.tmp.npz')
                np.savez(tmp, G=rr, means=means, quality=quality)
                tmp.replace(p)
                if key % 20 == 0:
                    print(f'Checkpoint chunk {key}', flush=True)
                if STOP:
                    pool.terminate()
                    raise SystemExit(143)
    arrays, qq = [], []
    for key, rr in chunks:
        with np.load(cache / f'{key:06d}.npz') as z:
            arrays.append(z['means'])
            qq.append(z['quality'])
    return np.concatenate(arrays), np.concatenate(qq)


def allocate_smoke(plan, limit, rng):
    if not limit:
        return plan
    strata = list(plan.groupby(['cohort', 'antibodies'], sort=True).indices.values())
    if limit < len(strata) * 3:
        raise ValueError('max-cells must allow at least three rows per cohort x antibody')
    chosen = []
    perm = [rng.permutation(v) for v in strata]
    for k in itertools.count():
        for v in perm:
            if k < len(v):
                chosen.append(int(v[k]))
                if len(chosen) == min(limit, len(plan)):
                    return plan.iloc[sorted(chosen)].reset_index(drop=True)


def make_plan(t, idx, goal, limit, seed):
    rng = np.random.default_rng(seed)
    records = []
    for direction, test_batch in [('b10', 0), ('b01', 1)]:
        # All masked-in GOAL cells, without additional histology/QC filtering.
        ev = np.flatnonzero((t['split'] == 2) & (t['batch'] == test_batch) & t['mask'][:, goal].any(1))
        for g in ev:
            records.append((int(g), direction, 'eval', idx.antibodies.iloc[g]))
        for ab, count in pd.Series(idx.antibodies.values[ev]).value_counts().items():
            candidates = np.flatnonzero((t['split'] == 0) & (t['hist_row'] < 0) &
                                       (t['batch'] == 1-test_batch) & (idx.antibodies.values == ab) &
                                       t['mask'][:, goal].any(1))
            selected = rng.choice(candidates, min(count, len(candidates)), replace=False)
            for g in selected:
                records.append((int(g), direction, 'train_matched', ab))
    plan = pd.DataFrame(records, columns=['G', 'direction', 'role', 'antibodies'])
    plan['cohort'] = plan.direction + ':' + plan.role
    plan = allocate_smoke(plan, limit, rng)
    # Each draw's split-0 rows occur in one training cohort, split-2 in one eval cohort.
    assert plan.G.is_unique
    return plan.sort_values('G').reset_index(drop=True)


def qp_unmix(x, spectra, offset, floors):
    """Exact 4-component nonnegative quadratic optimum by enumerating 16 active sets."""
    hess = spectra @ spectra.T
    q = x @ spectra.T - offset
    best = np.full(len(x), np.inf)
    ans = np.zeros_like(q)
    for bits in range(16):
        active = np.flatnonzero([(bits >> j) & 1 for j in range(4)])
        a = np.zeros_like(q)
        if len(active):
            a[:, active] = q[:, active] @ np.linalg.inv(hess[np.ix_(active, active)])
        valid = (a >= -1e-8).all(axis=1)
        objective = .5 * np.sum((a @ hess) * a, axis=1) - np.sum(a*q, axis=1)
        take = valid & (objective < best)
        ans[take], best[take] = a[take], objective[take]
    assert np.isfinite(best).all()
    return np.maximum(ans, floors)


def pool_model(sm, idx, t, batch, condition, corrected, raw):
    """Frozen authors' gate spectra; recover nuisance gradient offset on clean split-0 only.

    Authors' nonnegative L1 unmixing with fixed spectra has a quadratic KKT system.
    The offset absorbs their fitted condition background, positive channel shift and L1.
    Recovery is approximate because the released iterative solver stopped early.
    """
    spectra = []
    for group in GROUPS:
        components = []
        for marker in group.split('/'):
            gate = sm[f'selection_example_{marker}_positive_cells_batch_{batch}'].values.astype(bool)
            first = np.flatnonzero(gate)[0]
            ref = ((idx.antibodies.values == idx.antibodies.iloc[first]) &
                   (idx.batch.values == batch) &
                   (idx.slide_replicate.values == idx.slide_replicate.iloc[first]) & ~gate)
            # These are the authors' frozen spectrum gates, not a fit to evaluation outcomes.
            components.append(np.maximum(0, corrected[gate].mean(0) - corrected[ref].mean(0)))
        v = np.mean(components, axis=0)
        spectra.append(v / np.linalg.norm(v))
    spectra = np.asarray(spectra)
    cal = (t['split'] == 0) & (t['hist_row'] < 0) & (t['batch'] == batch)
    tr = cal & (idx.antibodies.values == condition)
    floors = np.nanmin(raw[cal], axis=0)
    active = raw > floors * (1 + 1e-6)
    a = np.where(active, raw, 0)
    residual = corrected @ spectra.T - a @ (spectra @ spectra.T)
    offset = []
    for j in range(4):
        rr = tr & active[:, j]
        if rr.sum() < 20:
            raise ValueError(f'Insufficient uncensored pooled calibration: {batch}/{condition}/{j}')
        offset.append(np.median(residual[rr, j]))
    return dict(spectra=spectra.tolist(), offset=offset, floors=floors.tolist())


def apply_pool(x, model):
    return qp_unmix(x, np.asarray(model['spectra']), np.asarray(model['offset']), np.asarray(model['floors']))


def curate(raw, rows, j, t, sm, keep_trend=False):
    """v3 cofactor/clean-split-0 z, frozen xfit correction, frozen v5 slide normalisation."""
    name = str(t['names'][j])
    y = np.full(len(rows), np.nan)
    for b in (0, 1):
        loc = t['batch'][rows] == b
        cal = (t['batch'] == b) & (t['split'] == 0) & (t['hist_row'] < 0)
        measured = t['mask'][:, j]
        if name.startswith('single:'):
            marker = name.split(':', 1)[1]
            original = sm[f'{marker}_single_antibody_model_unmixed'].values
            transformed = np.arcsinh(original[cal & measured] / COF[marker])
            v = np.arcsinh(raw[loc] / COF[marker])
        else:
            group = name.split(':', 1)[1]
            original = sm[f'{group}_full_model_unmixed'].values
            ref = cal & ~t['stained']
            med = np.median(original[ref])
            mad = np.median(np.abs(original[ref] - med))*1.4826 + 1e-6
            transformed = np.arcsinh((original[cal & measured] - med)/mad)
            v = np.arcsinh((raw[loc] - med)/mad)
        y[loc] = (v - transformed.mean()) / (transformed.std() + 1e-6)
    if not keep_trend:
        y -= (t['Y_v3'][rows, j] - t['Y_v4'][rows, j])
    if name.startswith('grp:'):
        for b in (0, 1):
            sel = t['mask'][:, j] & (t['batch'] == b)
            slides = np.unique(t['slide'][sel])
            if len(slides) < 2:
                continue
            cal = sel & (t['split'] == 0) & (t['hist_row'] < 0)
            stats = {}
            ref_parts = []
            for slide in slides:
                a = t['Y_v4'][cal & (t['slide'] == slide), j].astype(np.float64)
                q1, med, q3 = np.percentile(a, [25, 50, 75])
                scale = max((q3-q1)/1.349, 1e-6)
                stats[int(slide)] = (med, scale)
                ref_parts.append((a-med)/scale)
            ref = np.concatenate(ref_parts)
            for slide, (med, scale) in stats.items():
                loc = (t['batch'][rows] == b) & (t['slide'][rows] == slide)
                y[loc] = ((y[loc]-med)/scale-ref.mean())/ref.std()
    return y


def reliability(a, b):
    r = corr(a, b)
    rho = 2*r/(1+r) if np.isfinite(r) and r > -1 else float('nan')
    rmax = np.sqrt(rho) if np.isfinite(rho) and 0 <= rho <= 1 else float('nan')
    return r, rho, rmax


def fov_decomposition(y, p, fov):
    e = y-p
    _, inverse = np.unique(fov, return_inverse=True)
    n = np.bincount(inverse)
    means = np.bincount(inverse, weights=e)/n
    total = float(np.var(e))
    between = float(np.average((means-e.mean())**2, weights=n))
    within = float(np.mean((e-means[inverse])**2))
    assert np.isclose(total, between+within)
    # ANOVA random-intercept sensitivity; all FOVs including singleton groups contribute.
    nf, nc = len(n), len(e)
    msw = float(np.sum((e-means[inverse])**2)/(nc-nf)) if nc > nf else float('nan')
    msb = float(np.sum(n*(means-e.mean())**2)/(nf-1)) if nf > 1 else float('nan')
    n0 = (nc - np.sum(n*n)/nc)/(nf-1) if nf > 1 else float('nan')
    component = max(0., (msb-msw)/n0) if np.isfinite(msw+msb+n0) else float('nan')
    return dict(n=nc, n_fov=nf, singleton_fov=int((n == 1).sum()), residual_variance=total,
                variance_fov_means_unweighted=float(np.var(means)), between_variance_weighted=between,
                within_variance=within, fov_share=between/total if total else float('nan'),
                finite_n_null_share=(nf-1)/(nc-1) if nc > 1 else float('nan'),
                anova_fov_component=component, anova_share=component/(component+msw) if component+msw > 0 else float('nan'))


def predictions(source, direction, expected):
    result = []
    for arm in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland'):
        with np.load(source / 'runs' / f'{arm}_{direction}' / 'eval_preds.npz') as e:
            g, p = e['G'], e['P']
        if len(np.unique(g)) != len(g):
            raise ValueError(f'Duplicate prediction G in {arm}/{direction}')
        lookup = pd.Index(g).get_indexer(expected)
        if (lookup < 0).any():
            raise ValueError(f'Missing GOAL rows in {arm}/{direction}')
        result.append(p[lookup])
    return np.mean(result, axis=0)


def md_table(frame, columns=None):
    if columns is not None:
        frame = frame[columns]
    # No tabulate dependency in the training container.
    fmt = lambda v: f'{v:.5f}' if isinstance(v, (float, np.floating)) else str(v)
    return '| ' + ' | '.join(frame.columns) + ' |\n| ' + ' | '.join(['---']*len(frame.columns)) + ' |\n' + '\n'.join('| ' + ' | '.join(fmt(v) for v in row) + ' |' for row in frame.itertuples(index=False, name=None)) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=W)
    parser.add_argument('--out', type=Path, default=W/'runs/n064_ceiling')
    parser.add_argument('--maps', type=Path, default=N039)
    parser.add_argument('--workers', type=int, default=48)
    parser.add_argument('--chunk-size', type=int, default=128)
    parser.add_argument('--max-cells', type=int, default=0, help='0=full; smoke limits unique pixel rows globally')
    parser.add_argument('--seed', type=int, default=64)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_size < 1 or args.max_cells < 0:
        parser.error('workers/chunk-size must be positive; max-cells nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    inputs = [args.source/'targets_v5cx_v10.npz', args.source/'bsccm_sm.parquet', args.source/'bsccm_index.parquet', args.source/'pack/fluor.npy', args.maps]
    for d in ('b10', 'b01'):
        inputs += [args.source/'runs'/f'{arm}_{d}'/'eval_preds.npz' for arm in ('n048_v9_base', 'n048_v9_zoom2', 'n053_cytoland')]
    manifest = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), seed=args.seed,
                    max_cells=args.max_cells, chunk_size=args.chunk_size,
                    inputs=[dict(path=str(p.resolve()), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in inputs])
    mpth = args.out/'manifest.json'
    if mpth.exists() and json.loads(mpth.read_text()) != manifest:
        raise ValueError('Output directory belongs to different code/inputs/selection; use a new --out')
    atomic_json(mpth, manifest)
    if (args.out/'COMPLETE.json').exists():
        print('Already complete with identical manifest; no work needed', flush=True)
        return
    prereg = (Path(__file__).resolve().parent/'report.md').read_text().split('## Execution status')[0]
    (args.out/'preregistration.md').write_text(prereg)
    t0 = time.monotonic()
    sm = pd.read_parquet(args.source/'bsccm_sm.parquet')
    idx = pd.read_parquet(args.source/'bsccm_index.parquet')
    with np.load(inputs[0], allow_pickle=False) as z:
        t = {k: z[k] for k in z.files}
    if not np.array_equal(idx.global_index.values, np.arange(len(idx))) or not np.array_equal(sm.global_index.values, np.arange(len(idx))):
        raise ValueError('Global row identity mismatch')
    if not np.array_equal(idx.batch.values, t['batch']):
        raise ValueError('Draw identity mismatch')
    names = list(map(str, t['names']))
    goal = [j for j, n in enumerate(names) if n.startswith('single:') or (n.startswith('grp:') and n != 'grp:CD123/HLA-DR/CD14')]
    assert len(goal) == 11 and names[:8] == ['single:'+m for m in MARKERS]
    plan = make_plan(t, idx, goal, args.max_cells, args.seed)
    plan.to_csv(args.out/'cell_plan.csv', index=False)
    rows = plan.G.to_numpy(dtype=np.int64)
    signal.signal(signal.SIGTERM, stop_handler)
    signal.signal(signal.SIGUSR1, stop_handler)
    if STOP:
        raise SystemExit(143)
    pixel_start = time.monotonic()
    means, quality = pixel_cache(rows, args.out, args.source/'pack/fluor.npy', args.workers, args.chunk_size)
    pixel_seconds = time.monotonic()-pixel_start
    if STOP:
        raise SystemExit(143)
    # Global metadata maps native pack indices 5,4,3,2,1,0 to CHANNELS.
    # Checkpoints retain native channel order; reorder only after loading.
    means = means[..., ::-1]
    fg_n = np.array([a.sum() for a,b in PIX]); bg_n = np.array([b.sum() for a,b in PIX])
    fg_full = np.einsum('nhc,h->nc', means[:, :, 0], fg_n/fg_n.sum())
    bg_full = np.einsum('nhc,h->nc', means[:, :, 1], bg_n/bg_n.sum())
    released_fg = sm[[f'Fluor_{c}_total_raw' for c in CHANNELS]].values[rows]
    released_bg = sm[[f'Fluor_{c}_background' for c in CHANNELS]].values[rows]
    pixel_audit = []
    for c, channel in enumerate(CHANNELS):
        pixel_audit.append(dict(channel=channel, foreground_max_abs=float(np.max(np.abs(fg_full[:,c]-released_fg[:,c]))),
                                background_max_abs=float(np.max(np.abs(bg_full[:,c]-released_bg[:,c]))),
                                foreground_r=corr(fg_full[:,c],released_fg[:,c]), background_r=corr(bg_full[:,c],released_bg[:,c])))
    pd.DataFrame(pixel_audit).to_csv(args.out/'pixel_fidelity.csv', index=False)
    if max(a['foreground_max_abs'] for a in pixel_audit) > .01 or max(a['background_max_abs'] for a in pixel_audit) > .01:
        raise ValueError('Pixel mask or channel order does not reproduce released aggregates within .01 count')
    shading = sm[[f'Fluor_{c}_shading' for c in CHANNELS]].values
    numerator = sm[[f'Fluor_{c}_total_raw' for c in CHANNELS]].values - sm[[f'Fluor_{c}_background_lowess' for c in CHANNELS]].values
    xs = numerator/shading
    corrected = sm[[f'Fluor_{c}_shading_corrected' for c in CHANNELS]].values
    # Inverse the fixed authors channel scale; exactly relates N039 XS to their corrected channels.
    channel_scale = np.sum(xs*corrected, axis=0)/np.sum(xs*xs, axis=0)
    if not np.allclose(xs*channel_scale, corrected, rtol=1e-5, atol=1e-5):
        raise ValueError('Unexpected authors shading_corrected convention')
    half_x = (means[:,:,0]-means[:,:,1])/shading[rows,None,:]
    full_x = (fg_full-bg_full)/shading[rows]
    maps = json.loads(args.maps.read_text())['models']
    if any(v['method'] != 'recovered_clipped_affine' for v in maps.values()):
        raise ValueError('N039 fallback maps unsupported; inspect calibration')
    pooled_raw = sm[[g+'_full_model_unmixed' for g in GROUPS]].values
    models = {}
    fidelity = []
    for b in (0,1):
        model = pool_model(sm, idx, t, b, 'all', corrected, pooled_raw)
        models[f'all|b{b}'] = model
        # Independent split-2 map fidelity is diagnostic; no threshold tuning or map selection.
        for split in (0,2):
            rr = np.flatnonzero((t['batch']==b)&(idx.antibodies.values=='all')&(t['split']==split)&(t['hist_row']<0))
            pred = apply_pool(corrected[rr],model)
            for k,g in enumerate(GROUPS):
                fidelity.append(dict(target='grp:'+g,draw=b,split=split,n=len(rr),r=corr(pred[:,k],pooled_raw[rr,k]),
                                     residual_sd=float(np.std(pred[:,k]-pooled_raw[rr,k]))))
        for k,m in enumerate(MARKERS):
            model_s = maps[f'{m}|b{b}'];coef=np.asarray(model_s['coef'])
            for split in (0,2):
                rr=np.flatnonzero((t['batch']==b)&t['mask'][:,k]&(t['split']==split)&(t['hist_row']<0))
                pred=np.maximum(model_s['floor'],xs[rr]@coef[:6]+coef[6]);orig=sm[m+'_single_antibody_model_unmixed'].values[rr]
                fidelity.append(dict(target='single:'+m,draw=b,split=split,n=len(rr),r=corr(pred,orig),residual_sd=float(np.std(pred-orig))))
    atomic_json(args.out/'pooled_maps.json',json_clean(dict(channel_scale=channel_scale,models=models,
        source='authors compute_spectra, commit 64de2e0f681a09be3f2a91ae4ead27628bab353d; clean split-0 KKT nuisance offsets')))
    fid=pd.DataFrame(fidelity);fid.to_csv(args.out/'unmixing_fidelity.csv',index=False)
    goal_fid=fid[fid.target.isin([names[j] for j in goal])]
    fidelity_ok=bool((goal_fid.r>=.99).all())
    half_labels=np.full((len(rows),2,len(names)),np.nan);full_labels=np.full((len(rows),len(names)),np.nan)
    metrics=[]; curation=[]; reconstruction=[]
    for j in goal:
        valid=t['mask'][rows,j]; ii=np.flatnonzero(valid); gg=rows[ii]
        raw=np.full((len(ii),3),np.nan)
        for b in (0,1):
            sel=t['batch'][gg]==b; at=ii[sel]
            if j<8:
                model=maps[f'{MARKERS[j]}|b{b}'];coef=np.asarray(model['coef'])
                for h in (0,1):raw[sel,h]=np.maximum(model['floor'],half_x[at,h]@coef[:6]+coef[6])
                raw[sel,2]=np.maximum(model['floor'],full_x[at]@coef[:6]+coef[6])
            else:
                model=models[f'all|b{b}'];k=GROUPS.index(names[j].split(':',1)[1])
                for h in (0,1):raw[sel,h]=apply_pool(half_x[at,h]*channel_scale,model)[:,k]
                raw[sel,2]=apply_pool(full_x[at]*channel_scale,model)[:,k]
        for h in (0,1):half_labels[ii,h,j]=curate(raw[:,h],gg,j,t,sm)
        full_labels[ii,j]=curate(raw[:,2],gg,j,t,sm)
        original=sm[(MARKERS[j]+'_single_antibody_model_unmixed') if j<8 else (names[j].split(':',1)[1]+'_full_model_unmixed')].values[gg]
        replay=curate(original,gg,j,t,sm)
        curation.append(dict(target=names[j],n=len(gg),max_abs=float(np.max(np.abs(replay-t['Y'][gg,j]))),r=corr(replay,t['Y'][gg,j])))
        for (direction,role),sub in plan[valid].groupby(['direction','role']):
            take=sub.index.to_numpy(); r,rho,rmax=reliability(half_labels[take,0,j],half_labels[take,1,j])
            metrics.append(dict(direction=direction,draw=int(t['batch'][rows[take[0]]]),role=role,target=names[j],n=len(take),
                                split_half_r=r,rho=rho,r_max=rmax,half_sd_ratio=float(np.std(half_labels[take,0,j])/np.std(half_labels[take,1,j])) if np.std(half_labels[take,1,j]) else float('nan')))
            reconstruction.append(dict(direction=direction,role=role,target=names[j],n=len(take),
                 r_reconstructed_vs_v10=corr(full_labels[take,j],t['Y'][rows[take],j]),
                 reconstruction_error_sd=float(np.std(full_labels[take,j]-t['Y'][rows[take],j])),
                 half_difference_sd=float(np.std(half_labels[take,0,j]-half_labels[take,1,j]))))
    cur=pd.DataFrame(curation);cur.to_csv(args.out/'curation_replay.csv',index=False)
    if not (cur.max_abs < 2e-5).all():
        raise ValueError('Frozen v10 curation replay failed')
    met=pd.DataFrame(metrics);met.to_csv(args.out/'reliability.csv',index=False)
    rec=pd.DataFrame(reconstruction);rec.to_csv(args.out/'reconstruction_fidelity.csv',index=False)
    np.savez(args.out/'half_labels.npz',G=rows,A=half_labels[:,0],B=half_labels[:,1],full=full_labels,names=t['names'])
    summary=[]
    for (d,role),sub in met.groupby(['direction','role']):
        ok=(len(sub)==11) and np.isfinite(sub.r_max).all()
        summary.append(dict(direction=d,role=role,n_targets=len(sub),mean_r_max=float(sub.r_max.mean()) if ok else float('nan'),all_targets_defined=bool(ok)))
    summ=pd.DataFrame(summary);summ.to_csv(args.out/'summary.csv',index=False)
    fov=[]
    for d in ('b10','b01'):
        # Aggregate-only diagnostic: use ALL GOAL evaluation rows even in pixel smoke.
        test_batch=0 if d=='b10' else 1
        gg=np.flatnonzero((t['split']==2)&(t['batch']==test_batch)&t['mask'][:,goal].any(1))
        p=predictions(args.source,d,gg)
        expected_batch=0 if d=='b10' else 1
        assert np.all(t['split'][gg]==2) and np.all(t['batch'][gg]==expected_batch)
        for j in goal:
            ok=t['mask'][gg,j];rr=gg[ok]
            fov.append(dict(direction=d,target=names[j],ensemble_r=corr(t['Y'][rr,j],p[ok,j]),
                            **fov_decomposition(t['Y'][rr,j].astype(float),p[ok,j].astype(float),t['fov'][rr])))
    fd=pd.DataFrame(fov);fd.to_csv(args.out/'fov_error_share.csv',index=False)
    noise=[]
    # Actual target-file labels only: zero-filled masked marker entries are NEVER treated as controls.
    for b in (0,1):
        for split in (0,2):
            unst=(~t['stained'])&(t['batch']==b)&(t['split']==split)
            for j,name in enumerate(names):
                v=t['Y'][unst&t['mask'][:,j],j]
                noise.append(dict(draw=b,split=split,target=name,n=len(v),sd=float(np.std(v)) if len(v) else float('nan'),
                                  iqr=float(np.subtract(*np.percentile(v,[75,25]))) if len(v) else float('nan'),
                                  source='target-file masked-in labels'))
    nd=pd.DataFrame(noise);nd.to_csv(args.out/'unstained_spread.csv',index=False)
    # Raw released full-model outputs exist on unstained cells; their scale precedes v3-v5 curation.
    raw_noise=[]
    for b in (0,1):
        for split in (0,2):
            unst=(~t['stained'])&(t['batch']==b)&(t['split']==split)
            for k,g in enumerate(GROUPS):
                v=pooled_raw[unst,k];v=v[np.isfinite(v)]
                raw_noise.append(dict(draw=b,split=split,target='grp:'+g,n=len(v),raw_sd=float(np.std(v)),raw_iqr=float(np.subtract(*np.percentile(v,[75,25])))))
    pd.DataFrame(raw_noise).to_csv(args.out/'unstained_raw_pooled.csv',index=False)
    evaluation=met[met.role=='eval'];es=summ[summ.role=='eval']
    mean=float(es.mean_r_max.mean()) if np.isfinite(es.mean_r_max).all() and len(es)==2 else float('nan')
    focal=evaluation[evaluation.target.isin(['single:CD19','single:CD56'])]
    focal_ok=bool(len(focal)==4 and np.isfinite(focal.r_max).all() and (focal.r_max>=.75).all())
    registered_valid=(not args.max_cells) and fidelity_ok and np.isfinite(mean)
    decision=dict(full_run=not bool(args.max_cells),unmixing_fidelity_ok=fidelity_ok,goal_mean_r_max=mean,
                  registered_read_valid=bool(registered_valid),
                  registered_read_scope='pixel-noise of local-background reconstruction; not total v10 label reliability',
                  reconstructed_v10_r_min=float(rec[rec.role=='eval'].r_reconstructed_vs_v10.min()),
                  actual_v10_reachability_established=False,
                  near_ceiling=(mean<=.72) if registered_valid else None,
                  pixel_label_limited_refuted_CD19_CD56=focal_ok if registered_valid else None)
    elapsed=time.monotonic()-t0
    runtime=dict(unique_pixel_cells=len(rows),workers=args.workers,pixel_wall_seconds=pixel_seconds,total_wall_seconds=elapsed,
                 allocated_cpu_hours_at_48=elapsed*48/3600,foreground_counts=fg_n,background_counts=bg_n,
                 cells_with_zero_pixels=int((quality[:,0]>0).sum()),cells_with_ge4095_pixels=int((quality[:,1]>0).sum()),
                 calibration_scope='all clean split-0 aggregates; smoke caps only pixel rows; FOV and unstained aggregate diagnostics use all GOAL/control rows')
    atomic_json(args.out/'runtime.json',json_clean(runtime));atomic_json(args.out/'decision.json',json_clean(decision))
    status='FULL DATA' if not args.max_cells else 'SMOKE ONLY: no registered full-data conclusion'
    text=prereg+'\n## Results: '+status+'\n\n'
    text+='Directions b10 = train draw 1, evaluate draw 0; b01 = train draw 0, evaluate draw 1. Matched samples are clean split-0 cells in the training draw, matched without replacement by antibody condition to evaluation counts (shortages use all available cells). Smoke samples are balanced across cohort × antibody condition, capped globally.\n\n'
    text+='Each half uses its own foreground mean and corner-background mean, then the identical shading, unmixing, clipping, cofactors and frozen v10 curation. The xfit correction Y_v3 − Y_v4 is reused; no detrending is skipped or refitted. v5 slide medians/IQR and batch restandardisation are replayed from clean split-0 frozen Y_v4. Both halves share calibration and trend corrections, which can increase conditional agreement.\n\n'
    text+='Single-marker maps are the N039 clipped-affine recovery. Pooled maps use the [authors’ fixed gate spectra](https://github.com/Waller-Lab/BSCCM/blob/64de2e0f681a09be3f2a91ae4ead27628bab353d/bsccm/fluorescence_processing/demixing_util.py) and nonnegative active-set optimisation, with nuisance KKT offsets recovered on clean split-0 cells. This approximates the released iterative solver; full-data registered reads are withheld if any GOAL map has r < .99 on calibration or split-2 validation. No alternate unmixing is selected after evaluation.\n\n'
    text+=f'Pixel aggregates reproduce the [authors’ disk/corner geometry](https://github.com/Waller-Lab/BSCCM/blob/64de2e0f681a09be3f2a91ae4ead27628bab353d/bsccm/fluorescence_processing/compute_fluorescence.ipynb) within {max(max(a["foreground_max_abs"],a["background_max_abs"]) for a in pixel_audit):.6g} counts. Minimum GOAL unmixing fidelity r = {goal_fid.r.min():.6f}; maximum v10 curation replay error = {cur.max_abs.max():.3g}. Reconstructed full-label agreement with v10 is separately reported in reconstruction_fidelity.csv: local backgrounds can change labels despite high unmixing fidelity. Independent local backgrounds also add pixel variance compared with the authors’ smoothed backgrounds, which can lower this photon-noise estimate. Thus the upper-bound interpretation for total label reliability is heuristic, conditional on equivalent-half assumptions.\n\n'
    text+='**Estimand limitation:** independent local corner backgrounds replace the authors’ LOWESS backgrounds. High agreement of these halves is not sufficient to establish reliability of actual v10 labels. Read the full-label reconstruction audit below before interpreting the registered pixel-noise reads; no reachability claim is established.\n\n'
    text+=md_table(rec[rec.role=='eval'],['direction','target','r_reconstructed_vs_v10','reconstruction_error_sd','half_difference_sd'])+'\n'
    text+=md_table(met,['direction','role','target','n','split_half_r','rho','r_max'])+'\n'+md_table(summ)+'\n'
    text+=f'Equal-direction GOAL-style mean r_max = {mean:.6f}. Registered read valid: {registered_valid}. '
    if registered_valid:
        text+=f'Near-ceiling read (<= .72): {mean<=.72}. CD19/CD56 read (all four >= .75): {focal_ok}.\n\n'
    else:
        text+='The full-data registered reads remain pending.\n\n'
    text+='## FOV residual share\n\nThis aggregate diagnostic uses **all GOAL evaluation cells**, including in pixel smoke mode. Ensemble = arithmetic mean of n048_v9_base, n048_v9_zoom2 and n053_cytoland, joined by global G rather than array position. Labels are exactly v10 Y. The primary share is cell-weighted Var(E[residual|FOV])/Var(residual), using all measured cells, including singleton FOVs. Unweighted variance of FOV means, singleton counts, finite-sample null share and an ANOVA component sensitivity are in fov_error_share.csv. Sparse FOV means inflate the primary share; it must not be read as proven technical error. Predictions were trained with their original target scale; any scale mismatch is retained, with no evaluation-fitted rescaling.\n\n'
    text+=md_table(fd,['direction','target','n','n_fov','ensemble_r','fov_share','finite_n_null_share','anova_share'])+'\n'
    text+='## Unstained controls\n\nNo GOAL marker has masked-in labels on unstained cells in the targets file. Therefore a GOAL-scale unstained noise floor is unavailable; masked zero fills are not measurements. The available autofl spread is in unstained_spread.csv. Released raw pooled outputs on unstained cells are in unstained_raw_pooled.csv as an explicitly separate raw-scale diagnostic. These spreads include autofluorescence and biological variation. Aggregate diagnostics use all available control rows even in smoke mode; the pixel cap remains enforced.\n\n'
    text+=md_table(nd[nd.n>0],['draw','split','target','n','sd','iqr'])+'\n'
    text+=f'Runtime: {len(rows)} unique pixel cells, {pixel_seconds:.2f} s pixel stage, {elapsed:.2f} s total, {args.workers} workers. See runtime.json, manifest.json, cell_plan.csv, pixel_fidelity.csv, unmixing_fidelity.csv, curation_replay.csv and reconstruction_fidelity.csv. Completed chunks resume after preemption; changed code, inputs or selection require a fresh output directory.\n'
    (args.out/'report.md').write_text(text)
    if STOP:
        raise SystemExit(143)
    atomic_json(args.out/'COMPLETE.json',json_clean(dict(decision=decision,runtime=runtime)))
    print(md_table(summ),flush=True)
    print(json.dumps(json_clean(dict(decision=decision,runtime=runtime)),indent=2),flush=True)


if __name__=='__main__':
    main()
