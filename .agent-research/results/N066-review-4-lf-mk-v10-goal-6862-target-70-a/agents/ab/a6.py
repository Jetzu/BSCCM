import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'; T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M,b,S=T['Y'],T['mask'],T['batch'],T['slide']
names=[str(n) for n in T['names']]; j=names.index('grp:CD123/HLA-DR/CD14')
for bb in (0,1):
    m=M[:,j]&(b==bb)
    if m.sum(): 
        y=Y[m,j]; print('grp:CD123/HLA-DR/CD14 draw',bb,'n',m.sum(),'quantiles',np.round(np.percentile(y,[1,5,10,25,50,75,90,95,99]),2),'frac |y|<.05',round(float((np.abs(y)<.05).mean()),3))
        h,e=np.histogram(y,20); print('   hist',h.tolist(),np.round(e[[0,-1]],2))
# HLA-DR b01/b10 oracle with autofl output
pooled=[names.index(n) for n in ('grp:CD3/CD19/CD56','grp:CD45','grp:CD16')]
L=lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp,pa,d):
    a,bq=L(f'{tp}_{d}'),L(f'{pa}_{d}'); P=a['P'].copy(); P[:,pooled]=bq['P'][:,pooled]; return P,a['G']
for d in ('b10','b01'):
    base=np.mean([comp(f'n063_tpfov_s{s}',f'n063_panel_s{s}',d)[0] for s in (2,3)],0)
    z2,G=comp('n065_tpfov_z2','n065_panel_z2',d); cy,_=comp('n065_tpfov_cyto','n065_panel_cyto',d)
    P=np.mean([base,z2,cy]+[L(f'{r}_{d}')['P'] for r in ('n048_v9_base','n048_v9_zoom2','n053_cytoland')],0)
    ja=names.index('autofl')
    for t in ('single:HLA-DR','single:CD19','single:CD56','single:CD123','single:CD14'):
        jt=names.index(t); m=M[G,jt]; y=Y[G[m],jt]; X=np.c_[P[m,jt],P[m,ja],np.ones(m.sum())]
        beta=np.linalg.lstsq(X,y,rcond=None)[0]; r2=np.corrcoef(X@beta,y)[0,1]
        ya=Y[G[m],ja]  # true autofl label on the same cell (single slides carry autofl label)
        X3=np.c_[P[m,jt],ya,np.ones(m.sum())]; b3=np.linalg.lstsq(X3,y,rcond=None)[0]
        print(d,t,'r own',round(np.corrcoef(P[m,jt],y)[0,1],3),'oracle +pred autofl',round(r2,3),'coef',np.round(beta[:2],2),'| r(label, autofl LABEL)',round(np.corrcoef(y,ya)[0,1],3),'oracle +autofl label',round(np.corrcoef(X3@b3,y)[0,1],3),'| r(pred autofl, autofl label)',round(np.corrcoef(P[m,ja],ya)[0,1],3))
