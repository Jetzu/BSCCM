import numpy as np, pandas as pd
from scipy.stats import spearmanr, skew, kurtosis
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True)
Y,M,C,Tr=T['Y'],T['mask'],T['censored'],T['T']; names=[str(n) for n in T['names']]; sn=[str(s) for s in T['slide_names']]
sl,fov,sp=T['slide'],T['fov'],T['split']
def r(a,b): return np.corrcoef(a,b)[0,1]
z=np.load('/tmp/n095_ab/e8_b01.npz'); P,G=z['P'],z['G']; C3=np.load(W+'/runs/n048_v9_base_b01/eval_preds.npz')['C3']
print('C3',C3.shape, C3[:2])
for t in ('grp:CD45','grp:CD16','grp:CD3/CD19/CD56'):
    j=names.index(t)
    for s in (14,13):
        k=M[G,j]&(sl[G]==s); p=P[k,j]; y=Y[G[k],j]; c=C[G[k],j]
        A=np.c_[np.ones(len(p)),p]; yh=A@np.linalg.lstsq(A,y,rcond=None)[0]; res=y-yh
        o=np.argsort(-res**2); top=(res[o[:len(o)//50]]**2).sum()/(res**2).sum()
        # winsorized labels at 1/99 pct
        lo,hi=np.percentile(y,[1,99]); yw=np.clip(y,lo,hi)
        cl=C3[k].argmax(1) if C3.ndim==2 else C3[k]
        print(f'{t:18s} {sn[s][-17:]} n={len(p)} r={r(p,y):.3f} spearman={spearmanr(p,y)[0]:.3f} r_wins1-99={r(p,yw):.3f} top2%cells_SSEshare={top:.2f} skew={skew(y):.2f} kurt={kurtosis(y):.1f} q01/50/99={np.percentile(y,[1,50,99]).round(2)} cens={c.mean():.2f}')
        # per predicted class
        for q in range(3):
            m=cl==q
            if m.sum()>30: print(f'    cls{q} n={m.sum()} r_within={r(p[m],y[m]):.3f} ymean={y[m].mean():.2f} pmean={p[m].mean():.2f} ysd={y[m].std():.2f}')
