"""Evaluate arbitrary N080-format runs. Use LABEL=RUN_DIR to group folds into one arm."""
import argparse
from itertools import combinations
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

M = ['CD15', 'Siglec8', 'CD14', 'CD19', 'CD3', 'CD45', 'CD4', 'CD56', 'CD8']
ix = {m: i for i, m in enumerate(M)}

# Verbatim N080 scoring and hierarchical gates from N080-*/n080_eval.py.
def pr(a, b): a, b = a - a.mean(), b - b.mean(); d = np.sqrt((a * a).sum() * (b * b).sum()); return float((a * b).sum() / d) if d > 1e-12 else np.nan
def valley(x):
    g = GaussianMixture(2, random_state=0).fit(x[:, None]); mu = np.sort(g.means_.ravel())
    grid = np.linspace(mu[0], mu[1], 400); dens = np.exp(g.score_samples(grid[:, None]))
    return float(grid[np.argmin(dens)]) if mu[1] - mu[0] > 1e-6 else float(np.median(x))
def _hi(x, ref, k=5.0):   # threshold = median + k x 1.4826 MAD of a NEGATIVE reference population
    med = np.median(ref); return med + k * 1.4826 * np.median(np.abs(ref - med))
def gate(Iraw):
    """Hierarchical gating on a donor's OWN true labels (evaluation only; asinh-z values). Returns fine types.
    mono = CD14 > cut referenced on CD15-high cells (monocytes are CD15-dim); gran = CD15 valley among the rest; eos = Siglec8 > cut
    referenced on granulocytes (CD15-FITC spillover into Siglec8-PE); lymphocytes = rest; T = CD3 valley; B = CD19 > cut referenced on T;
    NK = CD56 valley among CD3- CD19- lymphocytes; T4/T8 by CD4/CD8 valleys within T. Gating scale asinh(I/1000)."""
    Yb = np.arcsinh(Iraw / 1000.0)   # gating scale: fixed noise-level cofactor (separates dim from bright; fitted cofactors squash CD15)
    x = lambda m: Yb[:, ix[m]]; n = len(Yb); t = np.array(['other'] * n, dtype=object)
    hiCD15 = x('CD15') > valley(x('CD15'))
    mono = x('CD14') > _hi(None, x('CD14')[hiCD15], 6.0)   # monocytes first: CD15-dim, CD14-bright (reference = CD15-high cells)
    gran = ~mono & (x('CD15') > valley(x('CD15')[~mono]))
    eos = gran & (x('Siglec8') > _hi(None, x('Siglec8')[gran], 4.0)); neut = gran & ~eos   # reference = neutrophil Siglec8 spillover
    lym = ~gran & ~mono
    T = lym & (x('CD3') > valley(x('CD3')[lym])) if lym.sum() > 50 else lym & False
    B = lym & ~T & (x('CD19') > _hi(None, x('CD19')[T], 5.0)) if T.sum() > 50 else lym & False
    rest = lym & ~T & ~B
    NK = rest & (x('CD56') > valley(x('CD56')[rest])) if rest.sum() > 50 else rest & False
    c4 = valley(x('CD4')[T]) if T.sum() > 50 else np.inf; c8 = valley(x('CD8')[T]) if T.sum() > 50 else np.inf
    t[neut] = 'neut'; t[eos] = 'eos'; t[mono] = 'mono'; t[lym] = 'lym_other'; t[B] = 'B'; t[NK] = 'NK'
    t[T] = 'Tother'; t[T & (x('CD4') > c4) & (x('CD8') <= c8)] = 'T4'; t[T & (x('CD8') > c8) & (x('CD4') <= c4)] = 'T8'
    return t
LYM = ('T4', 'T8', 'Tother', 'B', 'NK', 'lym_other'); PAIRS = [('T', 'CD4', 'CD8'), ('lymph', 'CD3', 'CD19'), ('lymph', 'CD3', 'CD56'), ('lymph', 'CD19', 'CD56')]


def registered_metric(metrics):
    table = metrics['model']['pairwise_correlation_bias']
    pairs = [f'{a}--{b}' for a, b in combinations(M, 2)]
    if set(table) != set(pairs):
        raise ValueError('Expected exactly the 36 canonical marker pairs in metrics.json.')
    values = np.asarray([table[k]['predicted_minus_true'] for k in pairs], dtype=float)
    valid = int(np.isfinite(values).sum())
    return dict(n_defined=valid, mean_abs_bias=float(np.abs(values).mean()) if valid == 36 else np.nan)


def fold_reference(raw):
    c = np.maximum(1.4826 * np.median(np.abs(raw - np.median(raw, axis=0)), axis=0), 1.0)
    transformed = np.arcsinh(raw / c)
    mean = transformed.mean(0)
    scale = transformed.std(0) + 1e-6
    y = (transformed - mean) / scale
    types = gate(raw)
    profiles = {k: y[types == k].mean(0) for k in np.unique(types)}
    return c, mean, scale, profiles, types


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', metavar='[LABEL=]RUN_DIR')
    parser.add_argument('--pack', type=Path, default='/hnvme/workspace/v181bb10-bsccm/external/sbiad452_pack')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    with np.load(args.pack / 'labels.npz', allow_pickle=False) as labels:
        if labels['names'].tolist() != M:
            raise ValueError('Unexpected marker order in pack.')
        raw = labels['I_raw'].astype(np.float64)
        samples = labels['sample'].astype(str)
    donors = np.array([int(re.fullmatch(r'sample([012])(?:_(?:early|late))?', s)[1]) for s in samples])
    folds, gates, seen = {}, {}, set()
    markers, within, pairs, pops, registered = [], [], [], [], []
    for spec in args.runs:
        label, sep, path = spec.partition('=')
        path = Path(path if sep else spec).resolve()
        label = label if sep else path.name
        with (path / 'metrics.json').open() as handle:
            metrics = json.load(handle)
        d = metrics['test_donor']
        if d not in (0, 1, 2) or metrics['names'] != M:
            raise ValueError(f'Unexpected donor or marker order in {path}/metrics.json')
        if (label, d) in seen:
            raise ValueError(f'Duplicate label and test donor {label}, {d}. Supply distinct LABEL= prefixes.')
        seen.add((label, d))
        if d not in folds:
            folds[d] = fold_reference(raw[donors == (d + 2) % 3])
        c, mean, scale, profiles, train_types = folds[d]
        base = dict(run=str(path), regime=label, test_donor=d)
        registered.append(dict(**base, **registered_metric(metrics)))
        with np.load(path / 'eval_preds.npz', allow_pickle=False) as pred:
            rows = pred['rows']
            if (pred['names'].tolist() != M or rows.shape != (metrics['n_test'],)
                or rows.dtype.kind not in 'iu' or len(np.unique(rows)) != len(rows)
                or (rows < 0).any() or (rows >= len(raw)).any() or not (donors[rows] == d).all()
                or not np.array_equal(pred['sample'], samples[rows])):
                raise ValueError(f'Row identity or marker order mismatch in {path}')
            y_a = pred['Y']
            y_b = (np.arcsinh(raw[rows] / c) - mean) / scale
            key = (d, rows.tobytes())
            if key not in gates:
                gates[key] = gate(raw[rows])
            types = gates[key]
            oracle = np.stack([profiles.get(k, np.zeros(9)) for k in types])
            for k in np.unique(types):
                pops.append(dict(**base, type=k, frac_test=(types == k).mean(),
                                 frac_train=(train_types == k).mean()))
            for name, field in [('model', 'P'), ('knn', 'P_knn')]:
                if name == 'knn' and field not in pred:
                    continue
                P = pred[field]
                if P.shape != y_b.shape or y_a.shape != y_b.shape or not np.isfinite([y_a, y_b, P]).all():
                    raise ValueError(f'Invalid prediction or truth matrix in {path}')
                arm = dict(**base, arm=f'{label}:{name}')
                for j, marker in enumerate(M):
                    r_b = pr(P[:, j], y_b[:, j])
                    reference = pr(oracle[:, j], y_b[:, j])
                    markers.append(dict(**arm, marker=marker, r_B=r_b, r_A=pr(P[:, j], y_a[:, j]),
                                        r_type_oracle=reference, model_minus_oracle=r_b - reference))
                    for k in np.unique(types):
                        s = types == k
                        if s.sum() >= 100 and y_b[s, j].std() > 1e-6:
                            within.append(dict(**arm, type=k, marker=marker, n=int(s.sum()),
                                               r_within=pr(P[s, j], y_b[s, j])))
                for group, a, b in PAIRS:
                    s = np.isin(types, ('T4', 'T8', 'Tother') if group == 'T' else LYM)
                    if s.sum() >= 100:
                        pairs.append(dict(**arm, group=group, pair=f'{a}-{b}', n=int(s.sum()),
                                          r_true=pr(y_b[s, ix[a]], y_b[s, ix[b]]),
                                          r_pred=pr(P[s, ix[a]], P[s, ix[b]])))
    arm_columns = ['run', 'regime', 'test_donor', 'arm']
    dm = pd.DataFrame(markers)
    dw = pd.DataFrame(within, columns=arm_columns + ['type', 'marker', 'n', 'r_within'])
    dp = pd.DataFrame(pairs, columns=arm_columns + ['group', 'pair', 'n', 'r_true', 'r_pred'])
    dp['pair_err'] = (dp.r_pred - dp.r_true).abs()
    dr = pd.DataFrame(registered)
    summary = dm.groupby('arm').agg(macro_r_B=('r_B', 'mean'), macro_r_A=('r_A', 'mean'),
                                    type_oracle=('r_type_oracle', 'mean'))
    summary['model_minus_oracle'] = summary.macro_r_B - summary.type_oracle
    summary['within_type_r'] = dw.groupby('arm').r_within.mean()
    summary['within_lymph_types_r'] = dw[dw.type.isin(LYM)].groupby('arm').r_within.mean()
    summary['pair_corr_abs_err'] = dp.groupby('arm').pair_err.mean()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, table in [('marker', dm), ('withintype', dw), ('pairs', dp), ('populations', pd.DataFrame(pops)),
                        ('36pairs', dr), ('summary', summary.reset_index())]:
        table.to_csv(args.out / f'n087_eval_{name}.csv', index=False)
    print(summary.round(4).to_string())


if __name__ == '__main__':
    main()
