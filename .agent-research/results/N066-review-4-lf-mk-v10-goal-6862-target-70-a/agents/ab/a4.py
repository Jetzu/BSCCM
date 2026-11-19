import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'; T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M,S,F=T['Y'],T['mask'],T['slide'],T['fov']
names=[str(n) for n in T['names']]; c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled=[names.index(n) for n in ('grp:CD3/CD19/CD56','grp:CD45','grp:CD16')]
L=lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp,pa,d):
    a,b=L(f'{tp}_{d}'),L(f'{pa}_{d}'); P=a['P'].copy(); P[:,pooled]=b['P'][:,pooled]; return P,a['G']
def pr(p,y,w):
    w=w/w.sum(); pm=(w*p).sum(); ym=(w*y).sum(); return (w*(p-pm)*(y-ym)).sum()/np.sqrt((w*(p-pm)**2).sum()*(w*(y-ym)**2).sum())
rng=np.random.default_rng(0); B=1000
sets={}
for d in ('b10','b01'):
    base=np.mean([comp(f'n063_tpfov_s{s}',f'n063_panel_s{s}',d)[0] for s in (2,3)],0)
    z2,G=comp('n065_tpfov_z2','n065_panel_z2',d); cy,_=comp('n065_tpfov_cyto','n065_panel_cyto',d)
    old=[L(f'{r}_{d}')['P'] for r in ('n048_v9_base','n048_v9_zoom2','n053_cytoland')]
    P6=np.mean([base,z2,cy]+old,0); P3=np.mean(old,0); Pc=np.mean([base,z2,cy],0)
    for j in c11:
        m=M[G,j]; g=G[m]; sets[(d,j)]=(P6[m,j],P3[m,j],Pc[m,j],Y[g,j],F[g])
# FOV bootstrap: resample FOVs within each test set (weights = multiplicity)
st=[]
for bi in range(B):
    vals=[]
    for k,(p6,p3,pc,y,f) in sets.items():
        u,inv=np.unique(f,return_inverse=True); cnt=np.bincount(rng.integers(0,len(u),len(u)),minlength=len(u)); w=cnt[inv].astype(float)
        vals.append((pr(p6,y,w),pr(p3,y,w),pr(pc,y,w)))
    v=np.array(vals).mean(0); st.append(v)
st=np.array(st)
pt=np.array([[np.corrcoef(p,y)[0,1] for p in (p6,p3,pc)] for (p6,p3,pc,y,f) in sets.values()]).mean(0)
print('point 6way/N053-3way/E3c',pt.round(4))
print('FOV-bootstrap SD of GOAL',st.std(0).round(4))
print('paired diff 6way-3way mean',(st[:,0]-st[:,1]).mean().round(4),'sd',(st[:,0]-st[:,1]).std().round(4),'P(<=0)',((st[:,0]-st[:,1])<=0).mean())
print('paired diff E3c-3way sd',(st[:,2]-st[:,1]).std().round(4))
print('n FOVs per set', {f'{d}:{names[j]}':len(np.unique(v[4])) for (d,j),v in sets.items()})
