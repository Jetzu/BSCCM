import numpy as np, json
W='/hnvme/workspace/v181bb10-bsccm'
def r(a,b): return np.corrcoef(a,b)[0,1]
T10=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); names=[str(n) for n in T10['names']]; sl=T10['slide']; b=T10['batch']
for run in ('n039_ceil_fov_v5_mt0','n039_ceil_fov_v5_mt128000'):
    z=np.load(f'{W}/runs/{run}/eval_preds.npz'); P,G=z['P'],z['G']
    res=json.load(open(f'{W}/runs/{run}/result.json')); print(run, {k:v for k,v in res.items() if not isinstance(v,(dict,list))} if isinstance(res,dict) else '')
    e=np.load('/tmp/n095_ab/e8_b01.npz'); Ge=e['G']; common=np.intersect1d(G,Ge); print(' n test',len(G),'draw1',(b[G]==1).sum(),'common with b01 test',len(common))
    pos={g:i for i,g in enumerate(G)}; pe={g:i for i,g in enumerate(Ge)}
    for tf in ('targets_v5.npz','targets_v5cx.npz','targets_v5cx_v10.npz'):
        T=np.load(f'{W}/{tf}',allow_pickle=True); Y,M=T['Y'],T['mask']
        out=[]
        for t in ('grp:CD45','grp:CD16','grp:CD3/CD19/CD56','single:HLA-DR'):
            j=names.index(t)
            for s in ((14,13) if t.startswith('grp') else (11,)):
                g=common[M[common,j]&(sl[common]==s)]
                ii=[pos[x] for x in g]; jj=[pe[x] for x in g]
                out.append(f'{t}@{s}: indist {r(P[ii,j],Y[g,j]):.3f} / E8 {r(e["P"][jj,j],Y[g,j]):.3f}')
        print(' ',tf,' | '.join(out))
