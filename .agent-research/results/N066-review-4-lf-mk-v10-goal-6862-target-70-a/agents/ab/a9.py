import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'; T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M,b,h=T['Y'],T['mask'],T['batch'],T['hist_row']
names=[str(n) for n in T['names']]; ja=names.index('autofl'); c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled=[names.index(n) for n in ('grp:CD3/CD19/CD56','grp:CD45','grp:CD16')]
L=lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp,pa,d):
    a,bq=L(f'{tp}_{d}'),L(f'{pa}_{d}'); P=a['P'].copy(); P[:,pooled]=bq['P'][:,pooled]; return P,a['G']
R=lambda p,y: np.corrcoef(p,y)[0,1]
cneg={}
for bb in (0,1):
    for j in range(8):
        m=M[:,j]&M[:,ja]&(b==bb)&(h<0); y=Y[m,j]; a=Y[m,ja]; k=y<np.median(y)
        cneg[(bb,j)]=np.cov(y[k],a[k])[0,1]/np.var(a[k])
print('neg-bulk leak coef (d0,d1):',{names[j][7:]:(round(cneg[(0,j)],2),round(cneg[(1,j)],2)) for j in range(8)})
tot=[]
for d,trd in (('b10',1),('b01',0)):
    base=np.mean([comp(f'n063_tpfov_s{s}',f'n063_panel_s{s}',d)[0] for s in (2,3)],0)
    z2,G=comp('n065_tpfov_z2','n065_panel_z2',d); cy,_=comp('n065_tpfov_cyto','n065_panel_cyto',d)
    P=np.mean([base,z2,cy]+[L(f'{r}_{d}')['P'] for r in ('n048_v9_base','n048_v9_zoom2','n053_cytoland')],0)
    rr=[]
    for j in c11:
        m=M[G,j]; y=Y[G[m],j]; p=P[m,j]
        if j<8: p2=p-cneg[(trd,j)]*P[m,ja]; rr.append((R(p,y),R(p2,y)))
        else: rr.append((R(p,y),R(p,y)))
    rr=np.array(rr); tot.append(rr.mean(0)); print(d,rr.mean(0).round(4),{names[j][7:]:round(rr[i,1]-rr[i,0],3) for i,j in enumerate(c11) if j<8})
print('GOAL',np.mean(tot,0).round(4))
