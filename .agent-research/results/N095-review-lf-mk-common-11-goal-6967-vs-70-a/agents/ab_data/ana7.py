import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'
def r(a,b): return np.corrcoef(a,b)[0,1]
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M=T['Y'],T['mask']; names=[str(n) for n in T['names']]
zi=np.load(f'{W}/runs/n039_ceil_fov_v5_mt0/eval_preds.npz'); Pi,Gi,Ci=zi['P'],zi['G'],zi['C3']; posi={g:i for i,g in enumerate(Gi)}
ix=names.index; heads=['single:HLA-DR','single:CD19','grp:CD3/CD19/CD56','single:CD3','single:CD56','single:CD123','single:CD14','single:CD45','single:CD16','autofl']
for d in ('b10','b01'):
    e=np.load(f'/tmp/n095_ab/e8_{d}.npz'); P,G=e['P'],e['G']; cls=Ci[[posi[g] for g in G]]
    for t in ('single:HLA-DR',):
        j=ix(t); k=M[G,j]; y=Y[G[k],j]; c=cls[k]
        print(d,t,'class frac',[round((c==q).mean(),2) for q in range(3)],'class y-means',[round(y[c==q].mean(),2) for q in range(3)],'class y-sd',[round(y[c==q].std(),2) for q in range(3)])
        for q in (0,2):
            m=c==q
            print(f'  cls{q} within-class r of E8 heads vs {t} label:', {h.split(":")[-1]:round(r(P[k,ix(h)][m],y[m]),2) for h in heads})
