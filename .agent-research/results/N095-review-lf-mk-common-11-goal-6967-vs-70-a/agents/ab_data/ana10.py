import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'
def r(a,b): return np.corrcoef(a,b)[0,1]
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]; ix=names.index
zi=np.load(f'{W}/runs/n039_ceil_fov_v5_mt0/eval_preds.npz'); posi={g:i for i,g in enumerate(zi['G'])}
for d in ('b10','b01'):
    e=np.load(f'/tmp/n095_ab/e8_{d}.npz'); P,G=e['P'],e['G']
    cls_own=np.load(f'{W}/runs/n048_v9_base_{d}/eval_preds.npz')['C3']; cls_in=zi['C3'][[posi[g] for g in G]]
    j=ix('single:HLA-DR'); k=M[G,j]; y=Y[G[k],j]; h=P[k,j]; c19=P[k,ix('single:CD19')]; c19=(c19-c19.mean())/c19.std()
    out=[]
    for name,cl in (('ownC3',cls_own[k]),('indistC3',cls_in[k])):
        L=(cl==0).astype(float)
        out.append(name+' '+' '.join(f'a{a}:{r(h+a*c19*L,y):.3f}' for a in (0,.1,.2,.3,.5)))
    print(d,'HLA-DR',' | '.join(out), '| agree C3 own vs indist', (cls_own[k]==cls_in[k]).mean().round(2))
