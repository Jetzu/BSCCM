"""N066: anisotropic part of the mean channel images (mean image minus its D4 average), from n066_d4_geometry.py's saved means;
which transform maps Bottom-half channels onto Right-half channels. writes n066_d4_aniso.csv"""
import numpy as np, pandas as pd, sys
sys.path.insert(0, '/home/hpc/v181bb/v181bb10/BSCCM'); from bsccm.data import LED
S = np.load('/tmp/n066_meanimgs.npz')['S']; ix = {n: i for i, n in enumerate(LED)}
TF = {'id': lambda x: x, 'transpose': lambda x: x.T, 'antitranspose': lambda x: np.rot90(x, 2).T, 'rot90': lambda x: np.rot90(x, 1),
      'rot270': lambda x: np.rot90(x, 3), 'rot180': lambda x: np.rot90(x, 2), 'flipud': lambda x: x[::-1], 'fliplr': lambda x: x[:, ::-1]}
A = lambda x: x - np.mean([f(x) for f in TF.values()], 0)
def r(a, b): a, b = a.ravel(), b.ravel(); return float(a @ b / np.sqrt((a @ a) * (b @ b)))
rows = []
pairs = [(f'DF_{a}_Bottom', f'DF_{a}_Right') for a in (50, 60, 70, 80)] + [('DPC_Bottom', 'DPC_Right'), ('DPC_Top', 'DPC_Left'), ('DPC_Bottom', 'DPC_Top'), ('DPC_Right', 'DPC_Left'), ('DPC_Bottom', 'DPC_Left'), ('DPC_Top', 'DPC_Right')]
for a_, b_ in pairs:
    for k, f in TF.items(): rows.append(dict(src=a_, dst=b_, transform=k, r=r(A(f(S[ix[a_]])), A(S[ix[b_]]))))
for n in ('LED119', 'Brightfield', 'DF_50', 'DF_90', 'DF_50_Bottom', 'DPC_Top'):
    rows.append(dict(src=n, dst=n, transform='aniso_share', r=float(np.linalg.norm(A(S[ix[n]])) / np.linalg.norm(S[ix[n]] - S[ix[n]].mean()))))
    for k, f in TF.items(): rows.append(dict(src=n, dst=n, transform='self_' + k, r=r(A(f(S[ix[n]])), A(S[ix[n]]))))
df = pd.DataFrame(rows); df.to_csv('n066_d4_aniso.csv', index=False); pd.set_option('display.width', 200)
print(df.pivot_table(index=['src', 'dst'], columns='transform', values='r').round(2).to_string())
