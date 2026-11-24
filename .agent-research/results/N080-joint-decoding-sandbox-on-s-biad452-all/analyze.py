import sys
import numpy as np
sys.path.insert(0, '/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N080-joint-decoding-sandbox-on-s-biad452-all/codex')
import train_ifc_sandbox as s
d = np.load('/tmp/n080_rt/fcs_I.npz'); I, S = d['I'], d['S']
M = s.MARKERS
c = np.maximum(1.4826*np.median(np.abs(I-np.median(I,0)),0),1.0)
Yt = np.arcsinh(I/c)
Y = np.zeros_like(Yt)
for sv in np.unique(S):
    m = S==sv; Y[m] = (Yt[m]-Yt[m].mean(0))/(Yt[m].std(0)+1e-6)
donor = s.donor_ids(S)
np.set_printoptions(precision=2, suppress=True, linewidth=200)
print('markers', M)
print('per-sample mean of asinh(I/c), in units of sample0 SD (what per-sample z removes):')
sd0 = Yt[S=='sample0'].std(0)
for sv in np.unique(S): print(f'{sv:14s}', (Yt[S==sv].mean(0)-Yt[S=='sample0'].mean(0))/sd0, ' sd ratio', Yt[S==sv].std(0)/sd0)
print('early-late shift within donor (units of donor pooled SD):')
for dn in (1,2):
    e, l = Yt[S==f'sample{dn}_early'], Yt[S==f'sample{dn}_late']
    print(dn, (l.mean(0)-e.mean(0))/Yt[donor==dn].std(0))
# Prospective alternative: z-score test donor with TRAINING donor stats vs per-sample own stats
# Gates: q75 positive fraction vs 2-comp GMM positive fraction
def gmm_thr(x, iters=200):
    lo, hi = np.quantile(x,[.1,.9]); mu = np.array([lo,hi]); sd = np.array([x.std()/2]*2); w = np.array([.5,.5])
    for _ in range(iters):
        p = w*np.exp(-.5*((x[:,None]-mu)/sd)**2)/sd; r = p/p.sum(1,keepdims=True)
        w = r.mean(0); mu = (r*x[:,None]).sum(0)/r.sum(0); sd = np.sqrt((r*(x[:,None]-mu)**2).sum(0)/r.sum(0))+1e-6
    o = np.argsort(mu); mu, sd, w = mu[o], sd[o], w[o]
    g = np.linspace(mu[0], mu[1], 2000)
    post = w[1]*np.exp(-.5*((g-mu[1])/sd[1])**2)/sd[1] > w[0]*np.exp(-.5*((g-mu[0])/sd[0])**2)/sd[0]
    return g[np.argmax(post)] if post.any() else mu[1], w[1], mu, sd
tr = donor==2; te = donor==0   # fold test-donor 0 -> train donor 2
gates = s.fit_gates(Y[tr])
print('q75 thr (train donor 2):', np.array(gates['positive']))
for j,m in enumerate(M):
    t, w1, mu, sd = gmm_thr(Y[tr, j])
    print(f'{m:8s} GMM thr {t:6.2f} GMM pos-frac {np.mean(Y[tr,j]>t):.3f} (w1 {w1:.3f}, mu {mu}, sd {sd})  q75 frac 0.25')
tg = s.gate_masks(Y[te], gates)
print('TEST donor0 true-label gate fractions with q75 gates:', {k: round(float(v.mean()),4) for k,v in tg.items()})
pops = s.coarse_populations(Y[te], gates); print({k: round(float(v.mean()),3) for k,v in pops.items()})
# DP with GMM gates fitted on train donor
thr = np.array([gmm_thr(Y[tr,j])[0] for j in range(9)])
pos = Y[te] > thr
print('DP CD3+CD19+ test: q75 gates', tg['CD3_CD19_DP'].mean(), ' GMM gates', (pos[:,M.index('CD3')]&pos[:,M.index('CD19')]).mean())
print('CD19+ frac test: GMM', pos[:,M.index('CD19')].mean(), 'CD3+ GMM', pos[:,M.index('CD3')].mean(), 'CD56+', pos[:,M.index('CD56')].mean())
# within q75-'CD3+CD19+' cells, how many are GMM-CD19+?
dp = tg['CD3_CD19_DP']; print('of q75-DP cells, frac GMM CD19+:', pos[dp, M.index('CD19')].mean(), ' GMM CD3+:', pos[dp, M.index('CD3')].mean())
# CD45 raw linear pooled target heavy tail
x = I[tr, M.index('CD45')]; z = (x-x.mean())/x.std()
print('raw CD45 train: skew', np.mean(z**3), 'kurt', np.mean(z**4), 'max z', z.max(), 'min z', z.min(), 'frac |z|>5', np.mean(np.abs(z)>5))
sm = I[tr][:, s.POOL_MEMBERS].sum(1); cc = s.robust_scale(sm); p1 = np.arcsinh(sm/cc); z1=(p1-p1.mean())/p1.std(); print('P1 kurt', np.mean(z1**4), 'c', cc)
