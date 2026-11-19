"""N066 (outsider idea 2): LED-channel geometry under the dihedral group D4. Mean flat-fielded cell image per channel over 3,000
random stained cells (r = (raw-30)/(bg-30)); cells are randomly oriented, so mean-image asymmetry comes from the illumination direction.
Tests: (1) which image transform maps DF_xx_Bottom onto DF_xx_Right and DPC pairs onto each other; (2) whether (full ring - Bottom half)
behaves like the mirror image of the Bottom half (= the missing Top half); (3) LED119 symmetry. Asymmetry score = centred mean image.
writes n066_d4_geometry.csv"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '/home/hpc/v181bb/v181bb10/BSCCM')
from bsccm.data import LED, W
led = np.load(W + '/pack/led_array.npy', mmap_mode='r'); bg = np.load(W + '/backgrounds_p50.npy', mmap_mode='r')
meta = np.load(W + '/cellmeta.npz'); py, px = meta['pos_y'], meta['pos_x']
T = np.load(W + '/targets_v5cx_v10.npz'); st = np.flatnonzero(T['stained'])
H = bg.shape[1]; ok = st[(py[st] >= 64) & (py[st] < H - 64) & (px[st] >= 64) & (px[st] < H - 64)]
rng = np.random.default_rng(0); G = np.sort(rng.choice(ok, 3000, replace=False))
S = np.zeros((23, 128, 128)); S2 = np.zeros((23, 128, 128))   # S: ratio images; S2: raw-30 (additive) images
for g in G:
    raw = np.asarray(led[g], np.float64); b = np.asarray(bg[:, py[g] - 64:py[g] + 64, px[g] - 64:px[g] + 64], np.float64)
    S += (raw - 30) / (b - 30); S2 += raw - 30
S /= len(G); S2 /= len(G)
c = lambda im: im - im.mean()
def r(a, b): a, b = c(a).ravel(), c(b).ravel(); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
TF = {'id': lambda x: x, 'transpose': lambda x: x.T, 'antitranspose': lambda x: np.rot90(x, 2).T, 'rot90': lambda x: np.rot90(x, 1),
      'rot270': lambda x: np.rot90(x, 3), 'rot180': lambda x: np.rot90(x, 2), 'flipud': lambda x: x[::-1], 'fliplr': lambda x: x[:, ::-1]}
ix = {n: i for i, n in enumerate(LED)}; rows = []
pairs = [(f'DF_{a}_Bottom', f'DF_{a}_Right') for a in (50, 60, 70, 80)] + [('DPC_Bottom', 'DPC_Right'), ('DPC_Top', 'DPC_Left'), ('DPC_Bottom', 'DPC_Top'), ('DPC_Right', 'DPC_Left')]
for a_, b_ in pairs:
    for k, f in TF.items(): rows.append(dict(test='map', src=a_, dst=b_, transform=k, r=r(f(S[ix[a_]]), S[ix[b_]])))
for a in (50, 60, 70, 80):   # derived Top = full - Bottom (raw-30 units), derived Left = full - Right
    top = S2[ix[f'DF_{a}']] - S2[ix[f'DF_{a}_Bottom']]; left = S2[ix[f'DF_{a}']] - S2[ix[f'DF_{a}_Right']]
    for k, f in TF.items():
        rows.append(dict(test='derived_top', src=f'DF_{a}_Bottom', dst=f'DF_{a}-Bottom', transform=k, r=r(f(S2[ix[f'DF_{a}_Bottom']]), top)))
        rows.append(dict(test='derived_left', src=f'DF_{a}_Right', dst=f'DF_{a}-Right', transform=k, r=r(f(S2[ix[f'DF_{a}_Right']]), left)))
    rows.append(dict(test='half_share', src=f'DF_{a}_Bottom', dst=f'DF_{a}', transform='mean ratio', r=float(S2[ix[f'DF_{a}_Bottom']].mean() / S2[ix[f'DF_{a}']].mean())))
    rows.append(dict(test='half_share', src=f'DF_{a}_Right', dst=f'DF_{a}', transform='mean ratio', r=float(S2[ix[f'DF_{a}_Right']].mean() / S2[ix[f'DF_{a}']].mean())))
for n in ('LED119', 'Brightfield', 'DF_50', 'DF_90'):
    for k, f in TF.items(): rows.append(dict(test='self_symmetry', src=n, dst=n, transform=k, r=r(f(S[ix[n]]), S[ix[n]])))
    rows.append(dict(test='asym_norm', src=n, dst=n, transform='sd(centred)/mean', r=float(c(S[ix[n]]).std() / S[ix[n]].mean())))
df = pd.DataFrame(rows); df.to_csv('n066_d4_geometry.csv', index=False)
best = df[df.test.isin(['map', 'derived_top', 'derived_left'])].sort_values('r', ascending=False).groupby(['test', 'src', 'dst']).head(2)
pd.set_option('display.width', 200); print(best.round(3).to_string(index=False)); print(df[df.test.isin(['half_share', 'self_symmetry', 'asym_norm'])].round(3).to_string(index=False))
np.savez('/tmp/n066_meanimgs.npz', S=S, S2=S2)
