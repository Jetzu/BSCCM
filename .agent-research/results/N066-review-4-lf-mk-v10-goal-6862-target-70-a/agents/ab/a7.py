import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'; T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M,b,h=T['Y'],T['mask'],T['batch'],T['hist_row']
names=[str(n) for n in T['names']]; ja=names.index('autofl'); c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled=[names.index(n) for n in ('grp:CD3/CD19/CD56','grp:CD45','grp:CD16')]
L=lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp,pa,d):
    a,bq=L(f'{tp}_{d}'),L(f'{pa}_{d}'); P=a['P'].copy(); P[:,pooled]=bq['P'][:,pooled]; return P,a['G']
R=lambda p,y: np.corrcoef(p,y)[0,1]
# training-draw label-only leak coefficient: v10 marker label ~ v10 autofl label on the same cells
coef={}
for bb in (0,1):
    for j in range(8):
        m=M[:,j]&M[:,ja]&(b==bb)&(h<0); y=Y[m,j]; a=Y[m,ja]
        coef[(bb,j)]=np.cov(y,a)[0,1]/np.var(a); 
print('label leak coef c_d = cov(marker,autofl)/var(autofl):'); print({names[j]:(round(coef[(0,j)],3),round(coef[(1,j)],3)) for j in range(8)})
print('r(marker label, autofl label) per draw:', {names[j]:tuple(round(R(Y[M[:,j]&M[:,ja]&(b==bb)&(h<0),j],Y[M[:,j]&M[:,ja]&(b==bb)&(h<0),ja]),3) for bb in (0,1)) for j in range(8)})
tot={}
for d,trd in (('b10',1),('b01',0)):
    base=np.mean([comp(f'n063_tpfov_s{s}',f'n063_panel_s{s}',d)[0] for s in (2,3)],0)
    z2,G=comp('n065_tpfov_z2','n065_panel_z2',d); cy,_=comp('n065_tpfov_cyto','n065_panel_cyto',d)
    P=np.mean([base,z2,cy]+[L(f'{r}_{d}')['P'] for r in ('n048_v9_base','n048_v9_zoom2','n053_cytoland')],0)
    rows={}
    for j in c11:
        m=M[G,j]; y=Y[G[m],j]; p=P[m,j]; r0=R(p,y)
        if j<8:
            pa=P[m,ja]; c=coef[(trd,j)]
            p_tr=p-c*pa                                   # remove the TRAINING-draw autofl leak (inductive, training labels only)
            bq=np.cov(p,pa)[0,1]/np.var(pa); p_perp=p-bq*pa  # label-free orthogonalisation on test predictions
            rows[names[j]]=(r0,R(p_tr,y),R(p_perp,y))
        else: rows[names[j]]=(r0,r0,r0)
    arr=np.array(list(rows.values())); tot[d]=arr.mean(0)
    print(d,'GOAL dir [as-is, minus train-draw leak, label-free orthogonal]',arr.mean(0).round(4))
    for k,v in rows.items():
        if k.startswith('single'): print('   ',k,np.round(v,3))
print('GOAL overall',((tot['b10']+tot['b01'])/2).round(4))
