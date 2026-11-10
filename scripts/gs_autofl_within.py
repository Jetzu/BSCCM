"""Giemsa-student autofl truth per arm: WITHIN-SLIDE Pearson on the anchored fov-holdout students' test_preds (immune to per-slide label
rescaling), cross-slide single-slide r from result.json, and MLL23 fine-contrast mean per seed (from a transfer_multiseed.csv). Long csv out."""
import argparse, json, csv, collections, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--anc', nargs='+', required=True, help='arm=run1,run2,..'); ap.add_argument('--xs', nargs='*', default=[], help='arm:dir=run (dir b1to0|b0to1)')
ap.add_argument('--transfer', default=''); ap.add_argument('--truth', default='targets_v4.npz'); ap.add_argument('--out', required=True); a = ap.parse_args()
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/' + a.truth, allow_pickle=True); ja = [str(n) for n in T['names']].index('autofl'); sl = T['slide']; sn = T['slide_names'].astype(str)
rows = []
for spec in a.anc:
    arm, runs = spec.split('=')
    for s, run in enumerate(runs.split(',')):
        E = np.load(f'{W}/runs/{run}/test_preds.npz'); g = E['gidx']; st = E['student'][:, ja]; y = T['Y'][g, ja]; m = T['mask'][g, ja]
        for k in np.unique(sl[g]):
            mm = m & (sl[g] == k); rows.append(dict(metric='gs_autofl_within_' + ('b0' if 'batch_0' in sn[k] else 'b1'), arm=arm, seed=s, value=float(np.corrcoef(st[mm], y[mm])[0, 1]), run=run))
for spec in a.xs:
    armdir, run = spec.split('='); arm, d = armdir.split(':'); f = json.load(open(f'{W}/runs/{run}/result.json'))['final']
    rows.append(dict(metric=f'gs_autofl_xslide_{d}', arm=arm, seed=0, value=f['autofl_truth_r/student_GS'], run=run))
if a.transfer:
    t = collections.defaultdict(list)
    for r in csv.DictReader(open(a.transfer)):
        if r['set'] == 'MLL23' and r['contrast'] not in ('segneut>lym CD16', 'mono>lym+neut CD14', 'lym>neut CD3'): t[(r['arm'], r['seed'])].append(float(r['auroc']))
    for (arm, s), v in t.items(): rows.append(dict(metric='mll23_fine_mean', arm=arm, seed=s, value=float(np.mean(v)), run=''))
with open(a.out, 'w') as fh: w = csv.DictWriter(fh, fieldnames=['metric', 'arm', 'seed', 'value', 'run']); w.writeheader(); w.writerows(rows)
agg = collections.defaultdict(list)
for r in rows: agg[(r['metric'], r['arm'])].append(r['value'])
for m in sorted({k[0] for k in agg}): print(f'{m:24s}', '  '.join(f'{arm} {np.mean(v):.3f}+-{np.std(v):.3f}(n{len(v)})' for (mm, arm), v in sorted(agg.items()) if mm == m))
