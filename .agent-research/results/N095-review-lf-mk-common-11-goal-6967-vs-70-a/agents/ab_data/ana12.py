import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M,C=T['Y'],T['mask'],T['censored']; names=[str(n) for n in T['names']]; sl=T['slide']; c3true=T['cls3']
zi=np.load(f'{W}/runs/n039_ceil_fov_v5_mt0/eval_preds.npz'); Gi,Ci=zi['G'],zi['C3']
for t in ('grp:CD45','grp:CD16','grp:CD3/CD19/CD56'):
    j=names.index(t)
    for s in (8,14,13):
        k=M[Gi,j]&(sl[Gi]==s); y=Y[Gi[k],j]; c=Ci[k]
        print(f'{t:18s} slide {s:2d} n={k.sum():5d} sd={y.std():.2f} cens={C[Gi[k],j].mean():.2f} within-type sd (lym/gran/mono)', [round(y[c==q].std(),2) for q in range(3)], 'type means',[round(y[c==q].mean(),2) for q in range(3)])
    # draw-0 all slide with TRUE gates (all splits)
    k=M[:,j]&(sl==8)&(c3true>=0); y=Y[k,j]; c=c3true[k]
    print(f'{t:18s} slide  8 TRUE gates all splits n={k.sum()} within-type sd', [round(y[c==q].std(),2) for q in range(3)])
