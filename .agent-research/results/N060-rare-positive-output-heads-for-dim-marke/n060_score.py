"""N060-N062 portfolio scoring (v10 GOAL, common-11, mean of b10/b01): per arm x seed single model; mean over seeds; GOAL without CD123;
arm as 3rd member with N048 base+zoom2 (seed-averaged member predictions). Bars (pre-registered): single mean > .6730 + .0054 = .6784
or 3-member > .6862 + .0054 = .6916. writes n060_scores.csv, n060_per_target.csv"""
import os, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]; c10 = [j for j in c11 if names[j] != 'single:CD123']
L = lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def rs(P, G, cols): return {names[j]: float(np.corrcoef(P[M[G, j], j], Y[G[M[G, j]], j])[0, 1]) for j in cols}
ARMS = {'r50 control': ['n051_v10_base', 'n051_v10_base_s1'], 'hlgauss': ['n060_hlgauss_s0', 'n060_hlgauss_s1'], 'twopart': ['n060_twopart_s0', 'n060_twopart_s1'],
        'pearson': ['n060_pearson_s0', 'n060_pearson_s1'], 'fovre': ['n061_fovre_s0', 'n061_fovre_s1'], 'pair': ['n061_pair_s0', 'n061_pair_s1'],
        'sampler': ['n061_sampler_s0', 'n061_sampler_s1'], 'autofl0': ['n061_autofl0_s0', 'n061_autofl0_s1'], 'panel': ['n062_panel_s0', 'n062_panel_s1']}
rows, pt = [], []
for arm, fams in ARMS.items():
    seeds_g, seeds_g10, mem = [], [], {}
    for f in fams:
        if not all(os.path.exists(f'{W}/runs/{f}_{d}/eval_preds.npz') for d in ('b10', 'b01')): continue
        r = {d: rs(L(f'{f}_{d}')['P'], L(f'{f}_{d}')['G'], c11) for d in ('b10', 'b01')}
        g = np.mean([np.mean(list(r[d].values())) for d in r]); g10 = np.mean([np.mean([v for k, v in r[d].items() if k != 'single:CD123']) for d in r])
        seeds_g.append(g); seeds_g10.append(g10); rows.append(dict(arm=arm, member=f, kind='single', goal=g, goal_noCD123=g10))
        for d in r:
            for k, v in r[d].items(): pt.append(dict(arm=arm, member=f, direction=d, target=k, r=v))
        for d in ('b10', 'b01'): mem.setdefault(d, []).append(L(f'{f}_{d}')['P'])
    if not seeds_g: continue
    rows.append(dict(arm=arm, member='mean', kind='single_mean', goal=np.mean(seeds_g), goal_noCD123=np.mean(seeds_g10), n_seeds=len(seeds_g)))
    e = []; e10 = []
    for d in ('b10', 'b01'):
        base = [L(f'n048_v9_base_{d}'), L(f'n048_v9_zoom2_{d}')]; G = base[0]['G']
        P = (base[0]['P'] + base[1]['P'] + np.mean(mem[d], 0)) / 3; r = rs(P, G, c11); e.append(np.mean(list(r.values()))); e10.append(np.mean([v for k, v in r.items() if k != 'single:CD123']))
    rows.append(dict(arm=arm, member='N048+arm', kind='ensemble3', goal=np.mean(e), goal_noCD123=np.mean(e10), n_seeds=len(seeds_g)))
df = pd.DataFrame(rows); df.to_csv('n060_scores.csv', index=False); pd.DataFrame(pt).to_csv('n060_per_target.csv', index=False)
print(df[df.kind != 'single'].round(4).to_string(index=False)); print(df[df.kind == 'single'].round(4).to_string(index=False))
