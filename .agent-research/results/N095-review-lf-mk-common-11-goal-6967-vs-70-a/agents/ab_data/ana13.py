import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]; sl=T['slide']
zi=np.load(f'{W}/runs/n039_ceil_fov_v5_mt0/eval_preds.npz'); Gi,Ci=zi['G'],zi['C3']
grp=['grp:CD3/CD19/CD56','grp:CD45','grp:CD16']; J=[names.index(g) for g in grp]
for s in (8,14,13):
    k=(sl[Gi]==s)&M[Gi][:,J].all(1); y=Y[Gi[k]][:,J]; c=Ci[k]
    out=[]
    for q,nm in enumerate(('lym','gran','mono')):
        R=np.corrcoef(y[c==q].T); out.append(f'{nm}: r(3/19/56,45)={R[0,1]:+.2f} r(3/19/56,16)={R[0,2]:+.2f} r(45,16)={R[1,2]:+.2f}')
    print('slide',s,' | '.join(out))
