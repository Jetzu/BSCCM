import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True)
Y,M,C=T['Y'],T['mask'],T['censored']; names=[str(n) for n in T['names']]; sn=[str(s) for s in T['slide_names']]
sl,b,sp=T['slide'],T['batch'],T['split']
for i,s in enumerate(sn):
    idx=sl==i; print(i,s,idx.sum(),'batch',np.unique(b[idx]),'split2',(sp[idx]==2).sum(),'targets',[names[j] for j in range(13) if M[idx,j].any()])
print()
print('per-target labelled cells: draw0 all/split2, draw1 all/split2, censored frac d0,d1')
for j,n in enumerate(names):
    r=[]
    for d in (0,1):
        k=M[:,j]&(b==d); r+= [k.sum(), (k&(sp==2)).sum()]
    c0=C[M[:,j]&(b==0),j].mean() if (M[:,j]&(b==0)).any() else np.nan
    c1=C[M[:,j]&(b==1),j].mean() if (M[:,j]&(b==1)).any() else np.nan
    print(f'{n:24s}',r, round(c0,3), round(c1,3))
