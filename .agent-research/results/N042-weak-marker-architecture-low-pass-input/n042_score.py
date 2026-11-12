"""N042 stage 1: low-pass (sigma 1.5 px) input probe vs the unblurred N031 base (same recipe), frozen v5 GOAL scorer, per target x direction.
Pre-registered: image-limited if >= 2 weak targets (CD19/CD56/CD123/HLA-DR) drop >= .05 while lineage markers (CD14/CD16/CD45) drop < .02."""
import sys, numpy as np, pandas as pd
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5.npz', allow_pickle=True); Y, M = T['Y'], T['mask']
names = [str(n) for n in T['names']]; c11 = [j for j, n in enumerate(names[:12]) if 'CD123/HLA' not in n]; rows = []
def load(r): E = np.load(f'{W}/runs/{r}/eval_preds.npz'); return E['P'], E['G']
for d in ('b10', 'b01'):
    for arm, run in (('base', f'n031_base_{d}'), ('lowpass1.5', f'n042_lp15_{d}')):
        P, G = load(run)
        for j in c11:
            m = M[G, j]; rows.append(dict(dir=d, arm=arm, target=names[j], r=float(np.corrcoef(P[m, j], Y[G[m], j])[0, 1])))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] + '/n042_lowpass_per_target.csv', index=False)
pv = df.pivot_table(index='target', columns=['dir', 'arm'], values='r'); 
for d in ('b10', 'b01'): pv[(d, 'drop')] = pv[(d, 'base')] - pv[(d, 'lowpass1.5')]
pd.set_option('display.width', 200); print(pv.round(3).to_string())
g = df.groupby(['arm', 'dir']).r.mean().unstack(); g['goal'] = g.mean(1); print(g.round(4))
weak = ['single:CD19', 'single:CD56', 'single:CD123', 'single:HLA-DR']; lin = ['single:CD14', 'single:CD16', 'single:CD45']
dr = {t: np.mean([pv.loc[t, (d, 'drop')] for d in ('b10', 'b01')]) for t in weak + lin}; print({k: round(v, 3) for k, v in dr.items()})
print('VERDICT image-limited:', sum(dr[t] >= .05 for t in weak) >= 2 and all(dr[t] < .02 for t in lin))
pd.DataFrame(dict(arm=df[df.arm == 'base'].assign(a='n031 base').a.tolist() + df[df.arm != 'base'].assign(a='low-pass 1.5px').a.tolist(),
                  value=df[df.arm == 'base'].r.tolist() + df[df.arm != 'base'].r.tolist(),
                  seed=(df[df.arm == 'base'].dir + ':' + df[df.arm == 'base'].target).tolist() + (df[df.arm != 'base'].dir + ':' + df[df.arm != 'base'].target).tolist())).to_csv(sys.argv[1] + '/plot_lowpass.csv', index=False)
