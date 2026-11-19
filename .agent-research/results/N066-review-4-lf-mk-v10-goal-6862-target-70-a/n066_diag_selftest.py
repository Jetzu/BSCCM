"""N068 pre-flight (written at review N066): self-test of the --diag-aug / --diag-tta transform AS IT IS WRITTEN in scripts/train_lf.py
(source block exec'd, not re-typed). Checks: involution; channel bookkeeping; on 1,500 real normalised cells (GPUNorm on CPU), the
anisotropic mean image of every swapped channel after diag() matches the untransformed partner channel. writes n066_diag_selftest.csv"""
import sys, re, numpy as np, torch, pandas as pd
sys.path.insert(0, '/home/hpc/v181bb/v181bb10/BSCCM'); from bsccm.data import chan_list, LED, W; from bsccm.gpu_norm import GPUNorm
src = open('/home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py').read()
blk = src[src.index('DIAG = None'):src.index('    return x.flip(-1).flip(-2).transpose(-1, -2)[:, DIAG]\n') + len('    return x.flip(-1).flip(-2).transpose(-1, -2)[:, DIAG]\n')]
class A: diag_aug = .5; diag_tta = 1; phase = 1
class AP:
    def error(self, m): raise SystemExit(m)
ns = dict(a=A, ap=AP(), torch=torch, CH=chan_list('no119'), DEVICE='cpu'); exec(blk, ns); diag, DIAG = ns['diag'], ns['DIAG']
x = torch.randn(4, 23, 128, 128); assert torch.equal(diag(diag(x)), x), 'not an involution'
nm = [LED[c] for c in chan_list('no119')] + ['phase']; perm = DIAG.tolist()
print('swaps:', {nm[k]: nm[v] for k, v in enumerate(perm) if k != v})
try:
    ns2 = dict(ns); ns2['CH'] = chan_list('all'); exec(blk, ns2); print('ERROR: LED119 accepted')
except SystemExit as e: print('LED119 correctly refused:', e)
led = np.load(W + '/pack/led_array.npy', mmap_mode='r'); dpc = np.load(W + '/pack/dpc.npy', mmap_mode='r')
T = np.load(W + '/targets_v5cx_v10.npz'); rng = np.random.default_rng(1); G = np.sort(rng.choice(np.flatnonzero(T['stained']), 1500, replace=False))
gn = GPUNorm('cpu'); S = torch.zeros(23, 128, 128); Sd = torch.zeros(23, 128, 128)
for i in range(0, len(G), 100):
    g = G[i:i + 100]; raw = torch.from_numpy(np.stack([np.array(led[k]).view(np.int16) for k in g])); ph = torch.from_numpy(np.stack([np.array(dpc[k]) for k in g]))
    xx = gn(raw, ph, torch.as_tensor(g), chan_list('no119'), True); S += xx.sum(0); Sd += diag(xx).sum(0)
S /= len(G); Sd /= len(G)
D4 = [lambda z: z, lambda z: z.T, lambda z: z.flip(0).flip(1).T, lambda z: z.flip(0), lambda z: z.flip(1), lambda z: z.flip(0).flip(1), lambda z: torch.rot90(z, 1), lambda z: torch.rot90(z, 3)]
an = lambda z: (z - torch.stack([f(z) for f in D4]).mean(0)).flatten()
cr = lambda u, v: float(u @ v / (u.norm() * v.norm()))
rows = []
for k, v in enumerate(perm[:-1]):
    if k == v: continue
    rows.append(dict(channel=nm[k], partner=nm[v], r_diag_vs_orig=cr(an(Sd[k]), an(S[k])), r_noswap_ctrl=cr(an(S[v].flip(0).flip(1).T), an(S[k]) * 0 + an(S[v])),
                     r_transform_only_no_channel_swap=cr(an(S[k].flip(0).flip(1).T), an(S[k]))))
df = pd.DataFrame(rows); df.to_csv(sys.argv[1] if len(sys.argv) > 1 else 'n066_diag_selftest.csv', index=False); print(df.round(3).to_string(index=False))
