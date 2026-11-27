#!/usr/bin/env python3
"""N092: field-of-view (FOV) shortcut audit of the N045 within-slide LF<->Giemsa retrieval.

The unmodified N045 runner (n045_sotalign.py) is imported by path. Its Data class, partitions, CCA,
raw-ridge recipe, SigLIP-pairs trainer (train_model) and retrieval() are reused as they are, so every trained
arm is an N045 v2 run (seed 0, 4,500 steps, 256 pairs/batch, dim 128, Lion lr 1e-4, wd 1e-5, eval/ckpt every 100).

Per protocol (fov_b0, fov_b1) and arm:
  1. train as N045 and dump the TEST-split embeddings of both modalities, with cell ids, slide and FOV, to npz;
  2. from the dumps only, compute
     (a) R@1 / R@10 over all test cells (N045 definition: GS queries, LF candidates, pessimistic ties);
     (b) FOV-hit@k: share of GS queries whose top-k LF candidates hold ANY cell of the query's own field, with its
         exact chance under random ranking (hypergeometric, per query);
     (c) within-field identification: for queries whose field has >= 2 test cells, accuracy of picking the true
         partner among that field's LF cells only (chance = mean 1/n_field);
     (d) R@k after removing all OTHER cells of the query's field from the candidate set;
     (e) field-mean R@k: each embedding replaced by its field's mean embedding, which measures what pure field
         identity yields for this embedding. Own-field cells tie, so the rank is >= n_field.
Arms: raw_ridge, cca, siglip_pairs (as N045) and two controls trained with the same SigLIP recipe:
  siglip_samefov_shuffled  each training positive = (GS of cell i, LF of a DIFFERENT cell j of i's field), a fixed
                           within-field cyclic derangement; cells of single-cell fields are dropped. Its best.pt is
                           selected on validation pairs shuffled the same way, so true pairing is never seen.
                           It is evaluated on the normal true-pair test set.
  siglip_pairs_multifov    true pairs on exactly the control's cells (single-cell fields dropped). This is the matched
                           reference for the control.
SigLIP arms are reported at best.pt (N045 rule: max validation R@10) and at last.pt (step 4,500, free of selection).
--reuse-n045 loads the existing N045 v2 artefacts (raw_ridge, cca, siglip_pairs) instead of training them.
--selftest checks the metric code on toys with known values, against a brute-force loop and against N045's retrieval().
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
N045_DIR = HERE.parent / 'N045-sotalign-semi-supervised-ot-alignment-gi'
N045_SCRIPT = N045_DIR / 'codex/scripts/n045_sotalign.py'
N045_V2_SUMMARY = N045_DIR / 'v2_summary.csv'
N045_V2_RUNS = Path('/home/hpc/v181bb/v181bb10/bsccm_runs/n045_v2')
DATA_ROOT = Path('/hnvme/workspace/v181bb10-bsccm')
ARMS = ('raw_ridge', 'cca', 'siglip_pairs', 'siglip_pairs_multifov', 'siglip_samefov_shuffled')
SIGLIP_ARMS = ('siglip_pairs', 'siglip_pairs_multifov', 'siglip_samefov_shuffled')
REUSABLE = ('raw_ridge', 'cca', 'siglip_pairs')
PROTOCOLS = ('fov_b0', 'fov_b1', 'cross_b0to1', 'cross_b1to0')
TOL = 1e-6   # N045 retrieval(): rank = #(sim >= sim_true - 1e-6), pessimistic ties


def load_n045(path=N045_SCRIPT):
    """Import the original N045 runner from its path (never modified)."""
    spec = importlib.util.spec_from_file_location('n045_sotalign', path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules['n045_sotalign'] = mod
    spec.loader.exec_module(mod)
    return mod


n045 = load_n045()          # sets N045's thread/env defaults before numpy/torch load
np, torch = n045.np, n045.torch
require, jsonable, atomic_json = n045.require, n045.jsonable, n045.atomic_json


# --------------------------------------------------------------------------------------------- metrics (numpy only)
def unit(x):
    x = np.asarray(x, np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def field_index(slide, fov):
    """Dense field id per cell from (slide, fov)."""
    key = np.stack([np.asarray(slide, np.int64), np.asarray(fov, np.int64)], 1)
    return np.unique(key, axis=0, return_inverse=True)[1].reshape(-1)


def hit_chance(n_cand, n_own, k):
    """P(a uniformly random ranking of n_cand candidates puts >= 1 of n_own given items in the top k)."""
    p_miss = np.ones(len(n_own))
    for t in range(min(k, n_cand)):
        p_miss *= np.clip((n_cand - n_own - t) / (n_cand - t), 0, None)
    return 1 - p_miss


def audit_metrics(gs, lf, field, ks=(1, 5, 10), tol=TOL):
    """Metrics (a)-(e) for row-paired test embeddings (row i of gs and lf is the same cell). GS = query."""
    gs, lf, field = unit(gs), unit(lf), np.asarray(field)
    n = len(gs)
    require(n > 1 and lf.shape == gs.shape and field.shape == (n,), 'audit_metrics needs paired arrays + fields')
    S = gs @ lf.T
    diag = np.diag(S).copy()
    same = field[:, None] == field[None, :]
    nf = same.sum(1)                                   # own-field cells in the gallery, incl. the true partner
    ge = S >= diag[:, None] - tol
    rank = ge.sum(1)                                   # (a)
    rank_in = (ge & same).sum(1)                       # (c) rank among own field only
    rank_ex = (ge & ~same).sum(1) + 1                  # (d) partner + other-field cells only
    best_own = np.where(same, S, -np.inf).max(1)
    fov_rank = (S >= best_own[:, None] - tol).sum(1)   # (b) pessimistic rank of the best own-field candidate
    uf, inv = np.unique(field, return_inverse=True)    # (e) field means per modality
    mx, my = np.zeros((len(uf), gs.shape[1])), np.zeros((len(uf), lf.shape[1]))
    np.add.at(mx, inv, gs); np.add.at(my, inv, lf)
    mx, my = unit(mx)[inv], unit(my)[inv]
    Sf = mx @ my.T
    rank_fm = (Sf >= np.diag(Sf)[:, None] - tol).sum(1)
    multi = nf >= 2
    m = dict(n_test=n, n_fields=len(uf), cells_per_field=n / len(uf), n_multicell_queries=int(multi.sum()),
             median_rank=float(np.median(rank)))
    for k in ks:
        m[f'R{k}'] = float(np.mean(rank <= k)); m[f'chance_R{k}'] = min(k / n, 1.)
        m[f'fovhit{k}'] = float(np.mean(fov_rank <= k)); m[f'chance_fovhit{k}'] = float(hit_chance(n, nf, k).mean())
        m[f'R{k}_excl_samefield'] = float(np.mean(rank_ex <= k))
        m[f'chance_R{k}_excl_samefield'] = float(np.minimum(k / (n - nf + 1), 1.).mean())
        m[f'fieldmean_R{k}'] = float(np.mean(rank_fm <= k))
    m['withinfield_acc'] = float(np.mean(rank_in[multi] == 1)) if multi.any() else None
    m['chance_withinfield_acc'] = float(np.mean(1 / nf[multi])) if multi.any() else None
    m['withinfield_acc_over_chance'] = (m['withinfield_acc'] / m['chance_withinfield_acc']) if multi.any() else None
    m['R10_singlecell_field_queries'] = float(np.mean(rank[~multi] <= 10)) if (~multi).any() else None
    m['R10_multicell_field_queries'] = float(np.mean(rank[multi] <= 10)) if multi.any() else None
    return m


def audit_metrics_bruteforce(gs, lf, field, k=10, tol=TOL):
    """Independent pure-Python loop version of the core metrics (selftest only)."""
    gs, lf = unit(gs).tolist(), unit(lf).tolist()
    field = list(np.asarray(field).tolist()); n = len(gs)
    dot = lambda a, b: sum(p * q for p, q in zip(a, b))
    S = [[dot(gs[i], lf[j]) for j in range(n)] for i in range(n)]
    r1 = rk = fh = ex = acc = acc_n = 0; chance_fh = chance_acc = 0.
    for i in range(n):
        own = [j for j in range(n) if field[j] == field[i]]
        rank = sum(1 for j in range(n) if S[i][j] >= S[i][i] - tol)
        r1 += rank <= 1; rk += rank <= k
        best = max(S[i][j] for j in own)
        fh += sum(1 for j in range(n) if S[i][j] >= best - tol) <= k
        ex += 1 + sum(1 for j in range(n) if field[j] != field[i] and S[i][j] >= S[i][i] - tol) <= k
        chance_fh += 1 - math.comb(n - len(own), k) / math.comb(n, k)
        if len(own) >= 2:
            acc_n += 1; chance_acc += 1 / len(own)
            acc += sum(1 for j in own if S[i][j] >= S[i][i] - tol) == 1
    return {'R1': r1 / n, f'R{k}': rk / n, f'fovhit{k}': fh / n, f'chance_fovhit{k}': chance_fh / n,
            f'R{k}_excl_samefield': ex / n, 'withinfield_acc': acc / acc_n, 'chance_withinfield_acc': chance_acc / acc_n}


def samefield_derangement(rows, field_of_row, seed):
    """Each kept row i gets a partner j != i of the same field (one random cycle per field); single-row fields dropped."""
    rows = np.asarray(rows, np.int64)
    f = field_of_row[rows]
    rng = np.random.default_rng(seed)
    kept, partner = [], []
    for g in np.unique(f):
        members = rows[f == g]
        if len(members) < 2:
            continue
        p = rng.permutation(members)
        kept.append(p); partner.append(np.roll(p, -1))
    require(len(kept) > 0, 'no multi-cell fields for the same-field control')
    kept, partner = np.concatenate(kept), np.concatenate(partner)
    o = np.argsort(kept)
    return kept[o], partner[o]


def multicell_rows(rows, field_of_row):
    rows = np.asarray(rows, np.int64)
    f = field_of_row[rows]
    u, c = np.unique(f, return_counts=True)
    return np.sort(rows[np.isin(f, u[c >= 2])])


# --------------------------------------------------------------------------------------------- selftest
def selftest():
    # Toy 1, analytic: fields of given size; 'correct' fields pair GS_i with LF_i, shuffled fields pair GS_i with an
    # LF carrying the code of the next cell of the field. Field code weight 1, cell code weight eps.
    eps, spec = .1, [(1, True), (1, True), (2, True), (2, False), (3, True), (3, False), (4, False), (12, False), (5, True)]
    n, F = sum(s for s, _ in spec), len(spec)
    gs, lf, field = np.zeros((n, F + n)), np.zeros((n, F + n)), np.zeros(n, np.int64)
    i0 = 0
    for f, (size, correct) in enumerate(spec):
        idx = np.arange(i0, i0 + size)
        gs[idx, f] = lf[idx, f] = 1.
        gs[idx, F + idx] = eps
        lf[idx, F + (idx if correct else np.roll(idx, -1))] = eps
        field[idx] = f; i0 += size
    m = audit_metrics(gs, lf, field)
    want = dict(R1=12/33, R10=21/33, median_rank=3., fovhit1=1., fovhit10=1., chance_fovhit1=213/1089,
                withinfield_acc=10/31, chance_withinfield_acc=7/31, R1_excl_samefield=1., R10_excl_samefield=1.,
                fieldmean_R1=2/33, fieldmean_R10=21/33, R10_singlecell_field_queries=1.,
                R10_multicell_field_queries=19/31, n_multicell_queries=31, n_fields=9)
    bad = {k: (m[k], v) for k, v in want.items() if abs(m[k] - v) > 1e-12}
    require(not bad, f'toy 1 analytic values wrong: {bad}')
    r = n045.retrieval(gs.astype(np.float32), lf.astype(np.float32))
    require(abs(r['R1'] - 12/33) < 1e-12 and abs(r['R10'] - 21/33) < 1e-12, f'toy 1 N045 retrieval {r}')
    # Toy 2, random: vectorised == brute force == N045 retrieval() on continuous data.
    rng = np.random.default_rng(0)
    sizes = rng.integers(1, 7, 45)
    field2 = np.repeat(np.arange(len(sizes)), sizes)
    fc = rng.normal(size=(len(sizes), 16))
    x = fc[field2] + .7 * rng.normal(size=(len(field2), 16))
    y = (fc[field2] + .7 * rng.normal(size=(len(field2), 16))) @ rng.normal(size=(16, 16)) * .3 + x * .5
    v, b = audit_metrics(x, y, field2), audit_metrics_bruteforce(x, y, field2)
    bad = {k: (v[k], b[k]) for k in b if abs(v[k] - b[k]) > 1e-12}
    require(not bad, f'toy 2 vectorised vs brute force: {bad}')
    r = n045.retrieval(x.astype(np.float32), y.astype(np.float32))
    require(abs(r['R1'] - v['R1']) < 1e-12 and abs(r['R10'] - v['R10']) < 1e-12, f'toy 2 vs N045 retrieval {r} {v}')
    # Derangement: no fixed points, same field, a permutation of the kept rows, singletons dropped.
    rows = np.arange(len(field2))
    kept, partner = samefield_derangement(rows, field2, 5)
    require(np.all(kept != partner) and np.array_equal(field2[kept], field2[partner]), 'derangement not same-field')
    require(np.array_equal(np.sort(partner), kept), 'derangement is not a permutation')
    require(np.array_equal(kept, multicell_rows(rows, field2)), 'derangement must keep exactly multi-cell fields')
    out = dict(toy1_n=n, toy1_R1=m['R1'], toy1_R10=m['R10'], toy1_withinfield_acc=m['withinfield_acc'],
               toy1_fovhit10=m['fovhit10'], toy1_fieldmean_R10=m['fieldmean_R10'],
               toy2_n=len(field2), toy2_R10=v['R10'], toy2_n045_R10=r['R10'], toy2_fovhit10=v['fovhit10'],
               toy2_withinfield_acc=v['withinfield_acc'], derangement_pairs=len(kept))
    print(json.dumps(dict(event='selftest_passed', **out)), flush=True)
    return out


# --------------------------------------------------------------------------------------------- training / dumping
class PartnerData:
    """Delegates to N045 Data; Y(rows) returns the LF features of each row's assigned partner (identity elsewhere)."""

    def __init__(self, d, partner_of):
        self._d, self._partner = d, partner_of

    def __getattr__(self, k):
        return getattr(self._d, k)

    def Y(self, rows):
        return self._d.Y(self._partner[np.asarray(rows, np.int64)])


def fov_subset(d, n_fov, seed):
    """Smoke universe: all paired cells of n_fov random fields per (draw, split) block."""
    t, h, rng, keep = d.t, d.h, np.random.default_rng(seed), []
    for b in (0, 1):
        for s in (0, 1, 2):
            rows = np.flatnonzero((t['batch'][h] == b) & (t['split'][h] == s))
            f = t['fov'][h[rows]]
            keep.append(rows[np.isin(f, rng.choice(np.unique(f), min(n_fov, len(np.unique(f))), replace=False))])
    return np.sort(np.concatenate(keep))


def alignment_layer(d, part, a, AlignmentLayer):
    """Same constructor call as n045.train_model for non-n009 arms."""
    dx, dy = d.G.shape[1], d.L.shape[1]
    return AlignmentLayer(vision_dimension=dy, text_dimension=dx, target_dimension=min(a.dim, len(part['train']) - 1, dy, dx),
                          linear_type='linear', cast_dtype=torch.float32, logit_scale=20., logit_bias=-10.,
                          width_factor=1.).to(a.device)


def load_state(model, path, a):
    ck = torch.load(path, map_location=a.device, weights_only=False)
    m = copy.deepcopy(model)
    m.load_state_dict(ck['model'])
    return n045.LearnedMap(m, a), int(ck['step'])


def fit_raw_ridge(d, part, a, folder):
    """N045 run() raw_ridge branch, same logic (that branch is inline in run(), so it cannot be called)."""
    import joblib
    scaler = n045.StandardScaler().fit(d.X(part['train']))
    xt, y = scaler.transform(d.X(part['train'])), d.Y(part['train'])
    best_r, best_head, meta = -1., None, {}
    for alpha in a.ridge_alphas:
        head = n045.Ridge(alpha=alpha).fit(xt, y)
        r = n045.retrieval(head.predict(scaler.transform(d.X(part['val']))), d.Y(part['val']), a.device)['R10']
        if r > best_r:
            best_r, best_head = r, head
            meta = dict(alpha=alpha, val_R10=r)
    mapping = n045.FrozenMap('raw_ridge', dict(scale=scaler, ridge=best_head))
    joblib.dump(mapping, folder / 'map.joblib')
    return mapping, meta


def train_arm(arm, d, part, a, folder, fld, ctx):
    """Returns ({ckpt: mapping}, meta). All fitting goes through the N045 functions."""
    MatchingModel, CCA, AlignmentLayer = ctx['upstream']
    if arm == 'raw_ridge':
        mapping, meta = fit_raw_ridge(d, part, a, folder)
        return {'best': mapping}, dict(meta, n_train=len(part['train']))
    if arm == 'cca':
        if ctx.get('cca') is None:
            ctx['cca'] = n045.fit_cca(d, part, a, CCA)
        return {'best': n045.FrozenMap('cca', dict(cca=ctx['cca']))}, dict(n_train=len(part['train']))
    dd, pp, extra = d, part, {}
    if arm == 'siglip_pairs_multifov':
        pp = dict(part, train=multicell_rows(part['train'], fld), val=multicell_rows(part['val'], fld))
    elif arm == 'siglip_samefov_shuffled':
        partner_of = np.arange(len(d.h))
        kt, pt = samefield_derangement(part['train'], fld, a.control_seed)
        partner_of[kt] = pt
        if a.control_selection == 'shuffled':
            kv, pv = samefield_derangement(part['val'], fld, a.control_seed + 1)
            partner_of[kv] = pv
        else:
            kv = part['val']
        require(np.all(partner_of[kt] != kt) and np.array_equal(fld[partner_of[kt]], fld[kt]), 'control pairing broken')
        np.savez(folder / 'control_pairs.npz', train_gs_row=kt, train_lf_row=partner_of[kt],
                 val_gs_row=kv, val_lf_row=partner_of[kv], selection=a.control_selection)
        dd, pp = PartnerData(d, partner_of), dict(part, train=kt, val=kv)
        extra = dict(control_selection=a.control_selection, control_seed=a.control_seed)
    mapping, meta = n045.train_model(dd, pp, a, 'siglip_pairs', a.seed_now, folder, ctx.get('cca'),
                                     MatchingModel, AlignmentLayer, None)
    last, step = load_state(mapping.model, folder / 'last.pt', a)
    require(step == a.steps, f'last.pt at step {step}, expected {a.steps}')
    meta = dict(best_step=meta['best_step'], best_val_R10=meta['best_val_R10'], n_train=len(pp['train']),
                n_val=len(pp['val']), **extra)
    return {'best': mapping, 'last': last}, meta


def reuse_arm(arm, d, part, a, root045, ctx):
    """Load an existing N045 v2 artefact instead of training (raw_ridge, cca, siglip_pairs)."""
    MatchingModel, CCA, AlignmentLayer = ctx['upstream']
    if arm == 'raw_ridge':
        import joblib
        sys.modules['__main__'].FrozenMap = n045.FrozenMap   # N045 pickled it from its own __main__
        sel = json.loads((root045 / 'raw_ridge/selection.json').read_text())
        return {'best': joblib.load(root045 / 'raw_ridge/map.joblib')}, dict(sel, n_train=len(part['train']), reused=True)
    if arm == 'cca':
        c = CCA.__new__(CCA)
        c.__dict__.update(torch.load(root045 / 'cca_teacher.pt', map_location=a.device, weights_only=False))
        return {'best': n045.FrozenMap('cca', dict(cca=c))}, dict(n_train=len(part['train']), reused=True)
    model = alignment_layer(d, part, a, AlignmentLayer)
    best, bstep = load_state(model, root045 / 'siglip_pairs/best.pt', a)
    last, lstep = load_state(model, root045 / 'siglip_pairs/last.pt', a)
    tr = json.loads((root045 / 'siglip_pairs/training.json').read_text())
    return {'best': best, 'last': last}, dict(best_step=bstep, last_step=lstep, best_val_R10=tr['best_val_R10'],
                                              n_train=len(part['train']), reused=True)


def dump_test(path, mapping, d, part, fld, a, info):
    te = part['test']
    zx, zy = mapping.encode(d.X(te), 'x'), mapping.encode(d.Y(te), 'y')
    r = n045.retrieval(zx, zy, a.device)    # N045's own number, stored for the reproduction check
    tmp = Path(str(path) + '.tmp.npz')
    np.savez_compressed(tmp, gs_emb=zx.astype(np.float32), lf_emb=zy.astype(np.float32), test_hist_row=te,
                        test_global_index=d.h[te], slide=d.t['slide'][d.h[te]], fov=d.t['fov'][d.h[te]],
                        batch=d.t['batch'][d.h[te]], field=fld[te], n045_R1=r['R1'], n045_R10=r['R10'],
                        info=json.dumps(jsonable(info)))
    os.replace(tmp, path)


def metrics_from_dump(path):
    z = np.load(path, allow_pickle=False)
    info = json.loads(str(z['info']))
    m = audit_metrics(z['gs_emb'], z['lf_emb'], z['field'])
    return dict(info, **m, n045_retrieval_R1=float(z['n045_R1']), n045_retrieval_R10=float(z['n045_R10']),
                dump=str(path))


def v2_reference(path=N045_V2_SUMMARY):
    ref = {}
    if path.exists():
        with path.open(newline='') as f:
            for r in csv.DictReader(f):
                if r['family'] == 'retrieval' and r['split'] == 'test' and r['variant'] == 'original' and r['metric'] in ('R1', 'R10'):
                    ref[(r['protocol'], r['arm'], r['metric'])] = float(r['mean'])
    return ref


def write_rows(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(jsonable(rows))
    os.replace(tmp, path)


def summarise(root, a, ref):
    rows = []
    for p in sorted((root / 'dumps').glob('*.npz')):
        m = metrics_from_dump(p)
        for k in ('R1', 'R10'):
            v = None if m['smoke'] else ref.get((m['protocol'], m['arm'], k))
            m[f'n045_v2_{k}'] = v
            m[f'delta_vs_v2_{k}'] = (m[k] - v) if v is not None else None
        rows.append(m)
    by = {(r['arm'], r['ckpt']): r for r in rows}
    for r in rows:   # headline ratio: control (or any arm) vs the real SigLIP at the same checkpoint rule
        base = by.get(('siglip_pairs', r['ckpt'])) or by.get(('siglip_pairs', 'best'))
        r['R10_over_siglip_pairs'] = r['R10'] / base['R10'] if base and base['R10'] else None
    write_rows(root / 'metrics.csv', rows)
    atomic_json(root / 'metrics.json', rows)
    return rows


def print_table(rows):
    cols = dict(arm='arm', ckpt='ckpt', R1='R1', R10='R10', n045_v2_R10='v2_R10', fovhit10='fovhit10',
                chance_fovhit10='ch_fovhit10', withinfield_acc='inField_acc', chance_withinfield_acc='ch_inField',
                R10_excl_samefield='R10_exclFOV', chance_R10_excl_samefield='ch_R10_excl',
                fieldmean_R10='fieldmean_R10', n_train='n_train')
    print(f'{"arm":>24} ' + ' '.join(f'{h:>13}' for c, h in cols.items() if c != 'arm'))
    for r in rows:
        cells = [f'{r.get(c):>13.4f}' if isinstance(r.get(c), float) else f'{str(r.get(c)):>13}' for c in cols if c != 'arm']
        print(f'{r["arm"]:>24} ' + ' '.join(cells))
    sys.stdout.flush()


def run(a):
    out = a.out
    out.mkdir(parents=True, exist_ok=True)
    for bad in (a.data_root.resolve(), a.sotalign_root.resolve()):
        require(not out.resolve().is_relative_to(bad), 'output must be outside the data/source trees')
    upstream = n045.upstream(a.sotalign_root)
    torch.set_num_threads(a.threads)
    torch.use_deterministic_algorithms(True)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in ('resume', 'fingerprint', 'seed_now')}
    cfg['n045_runner_sha256'] = n045.digest(N045_SCRIPT)
    cfg['n092_audit_sha256'] = n045.digest(__file__)
    cfg['environment'] = dict(torch=torch.__version__, numpy=np.__version__, device=a.device)
    a.fingerprint = hashlib.sha256(json.dumps(jsonable(cfg), sort_keys=True).encode()).hexdigest()
    cpath = out / 'config.json'
    if cpath.exists():
        old = json.loads(cpath.read_text())
        require(a.resume and old['fingerprint'] == a.fingerprint,
                'output exists with another config: use --resume with the identical command, or a new --out')
    atomic_json(cpath, dict(config=cfg, fingerprint=a.fingerprint))
    t0 = time.time()
    d = n045.Data(a)
    if a.smoke_fovs:
        d.universe = fov_subset(d, a.smoke_fovs, 90)
    fld = field_index(d.t['slide'][d.h], d.t['fov'][d.h])    # field id per paired (hist) row
    print(json.dumps(dict(event='data_loaded', seconds=round(time.time() - t0, 1), n_pairs=len(d.h),
                          universe=len(d.universe))), flush=True)
    ref = v2_reference()
    all_rows = []
    for protocol in a.protocols:
        for seed in a.seeds:
            a.seed_now = seed
            part = d.partition(protocol, seed)
            root = out / protocol / f'seed{seed}'
            (root / 'dumps').mkdir(parents=True, exist_ok=True)
            root045 = N045_V2_RUNS / f'original_{protocol}' / protocol / f'seed{seed}'
            if not a.smoke_fovs and (root045 / 'partitions.npz').exists():
                with np.load(root045 / 'partitions.npz') as p0:
                    for k in ('train', 'val', 'test'):
                        require(np.array_equal(p0[k], part[k]), f'{protocol} {k} partition differs from N045 v2')
                print(json.dumps(dict(event='partition_matches_n045_v2', protocol=protocol, seed=seed)), flush=True)
            np.savez(root / 'partitions.npz', train=part['train'], val=part['val'], test=part['test'],
                     field_train=fld[part['train']], field_val=fld[part['val']], field_test=fld[part['test']])
            ctx = dict(upstream=upstream, cca=None)
            for arm in a.arms:
                done = sorted((root / 'dumps').glob(f'{arm}__*.npz'))
                if a.resume and done:
                    continue
                if n045.STOP:
                    raise SystemExit(143)
                folder = root / arm
                folder.mkdir(exist_ok=True)
                ta = time.time()
                if a.reuse_n045:
                    if arm not in REUSABLE:
                        print(json.dumps(dict(event='skip_not_reusable', arm=arm)), flush=True)
                        continue
                    maps, meta = reuse_arm(arm, d, part, a, root045, ctx)
                else:
                    maps, meta = train_arm(arm, d, part, a, folder, fld, ctx)
                meta['fit_seconds'] = round(time.time() - ta, 1)
                atomic_json(folder / 'audit_selection.json', meta)
                for ckpt, mapping in maps.items():
                    info = dict(protocol=protocol, seed=seed, arm=arm, ckpt=ckpt, smoke=bool(a.smoke_fovs),
                                reused_n045=bool(a.reuse_n045), steps=a.steps, **meta)
                    dump_test(root / 'dumps' / f'{arm}__{ckpt}.npz', mapping, d, part, fld, a, info)
                print(json.dumps(dict(event='arm_complete', protocol=protocol, seed=seed, arm=arm, **meta)), flush=True)
            rows = summarise(root, a, ref)
            all_rows.extend(rows)
            print(f'== {protocol} seed{seed} (GS query -> LF gallery, n_test={len(part["test"])})')
            print_table(rows)
            for r in rows:
                if not a.smoke_fovs and r['ckpt'] == 'best' and r.get('delta_vs_v2_R10') is not None and not a.reuse_n045:
                    flag = 'OK' if abs(r['delta_vs_v2_R10']) <= .02 else 'CHECK'
                    print(f'reproduction {protocol} {r["arm"]}: R10 {r["R10"]:.4f} vs N045 v2 {r["n045_v2_R10"]:.4f} [{flag}]')
            if a.results_dir:
                tag = f'{protocol}_seed{seed}' + ('_smoke' if a.smoke_fovs else '') + ('_reuse045' if a.reuse_n045 else '')
                write_rows(a.results_dir / f'metrics_{tag}.csv', rows)
                atomic_json(a.results_dir / f'metrics_{tag}.json', rows)
    atomic_json(out / 'completion.json', dict(completed=True, seconds=round(time.time() - t0, 1),
                                               protocols=a.protocols, seeds=a.seeds, arms=a.arms))
    return all_rows


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path)
    p.add_argument('--results-dir', type=Path, help='also copy metrics_<protocol>_seed<seed>.csv/json here')
    p.add_argument('--selftest', action='store_true', help='metric toys only, no data, then exit')
    p.add_argument('--from-dumps', action='store_true', help='recompute metrics from existing --out dumps only')
    p.add_argument('--reuse-n045', action='store_true', help='load N045 v2 artefacts instead of training (no controls)')
    p.add_argument('--protocols', nargs='+', choices=PROTOCOLS, default=['fov_b0', 'fov_b1'])
    p.add_argument('--arms', nargs='+', choices=ARMS, default=list(ARMS))
    p.add_argument('--seeds', nargs='+', type=int, default=[0])
    p.add_argument('--control-selection', choices=['shuffled', 'true'], default='shuffled',
                   help='validation pairs used to pick the control best.pt (shuffled = never sees true pairs)')
    p.add_argument('--control-seed', type=int, default=9200)
    p.add_argument('--smoke-fovs', type=int, default=0, help='>0: universe = all cells of N random fields per block')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--threads', type=int, default=48)
    # N045 v2 settings (n045_cpu.sbatch); N045 defaults for everything not set there.
    p.add_argument('--steps', type=int, default=4500)
    p.add_argument('--batch-pairs', type=int, default=256)
    p.add_argument('--batch-unpaired', type=int, default=256)
    p.add_argument('--dim', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--wd', type=float, default=1e-5)
    p.add_argument('--optimizer', choices=['lion', 'adamw'], default='lion')
    p.add_argument('--cca-lam', type=float, default=.1)
    p.add_argument('--n-unpaired', type=int, default=100000)
    p.add_argument('--head-train', type=int, default=100000)
    p.add_argument('--head-val', type=int, default=10000)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--ridge-alphas', nargs='+', type=float, default=[1., 10., 100., 1000.])
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)
    # Fields n045.Data / partition / train_model read; fixed at the N045 v2 values (unused by these arms or no-ops).
    a.data_root, a.sotalign_root = DATA_ROOT, DATA_ROOT / 'external/SOTAlign'
    a.lf_dir = DATA_ROOT / 'runs/n029_r18_v4c_fov/probe'
    a.giemsa, a.pbc = DATA_ROOT / 'emb/giemsa_dinobloom_B.npy', DATA_ROOT / 'emb/pbc_dinobloom_B.npy'
    a.targets, a.legacy_truth = DATA_ROOT / 'targets_v5c.npz', DATA_ROOT / 'gs_autofl_detrended.npy'
    a.smoke, a.max_pbc, a.max_pairs, a.max_val_pairs = False, 0, 0, 0
    a.allow_unverified_lf_provenance = False
    a.klot_weight, a.eps_student, a.eps_teacher, a.sinkhorn_iters, a.sail_negatives = .001, .05, .01, 100, 'expanded'
    a.min_head_labels = 30
    if not a.selftest:
        require(a.out is not None, '--out is required')
        a.out = a.out.resolve()
    if a.results_dir:
        a.results_dir = a.results_dir.resolve()
    require(a.steps % a.eval_every == 0, 'steps must be a multiple of eval-every so last.pt is the final step')
    return a


def main():
    a = parse_args()
    if a.selftest:
        selftest()
        return
    selftest()   # cheap; refuse to produce numbers with broken metric code
    if a.from_dumps:
        ref = v2_reference()
        for protocol in a.protocols:
            for seed in a.seeds:
                root = a.out / protocol / f'seed{seed}'
                rows = summarise(root, a, ref)
                print(f'== {protocol} seed{seed} (from dumps)'); print_table(rows)
                if a.results_dir:
                    write_rows(a.results_dir / f'metrics_{protocol}_seed{seed}.csv', rows)
        return
    signal.signal(signal.SIGTERM, n045.stop_handler)
    signal.signal(signal.SIGUSR1, n045.stop_handler)
    with n045.threadpool_limits(limits=a.threads):
        run(a)


if __name__ == '__main__':
    main()
