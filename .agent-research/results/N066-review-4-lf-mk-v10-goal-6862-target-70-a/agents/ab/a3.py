import numpy as np, pandas as pd
from scipy.stats import spearmanr
W='/hnvme/workspace/v181bb10-bsccm'; T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M,S=T['Y'],T['mask'],T['slide']; SN=T['slide_names'].astype(str)
names=[str(n) for n in T['names']]; c11=[j for j,n in enumerate(names[:12]) if 'CD123/HLA' not in n]
pooled=[names.index(n) for n in ('grp:CD3/CD19/CD56','grp:CD45','grp:CD16')]
L=lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp,pa,d):
    a,b=L(f'{tp}_{d}'),L(f'{pa}_{d}'); P=a['P'].copy(); P[:,pooled]=b['P'][:,pooled]; return P,a['G']
ix=names.index
R=lambda p,y: np.corrcoef(p,y)[0,1]
z=lambda v:(v-v.mean())/v.std()
out={}
for d in ('b10','b01'):
    base=np.mean([comp(f'n063_tpfov_s{s}',f'n063_panel_s{s}',d)[0] for s in (2,3)],0)
    z2,G=comp('n065_tpfov_z2','n065_panel_z2',d); cy,_=comp('n065_tpfov_cyto','n065_panel_cyto',d)
    P=np.mean([base,z2,cy]+[L(f'{r}_{d}')['P'] for r in ('n048_v9_base','n048_v9_zoom2','n053_cytoland')],0)
    out[d]=(P,G)
    rs={}
    for j in c11:
        m=M[G,j]; rs[names[j]]=R(P[m,j],Y[G[m],j])
    print(d,'GOAL dir',round(np.mean(list(rs.values())),4))
    # per-slide prediction standardisation (label-free) for pooled targets
    for mode in ('center','robustz','z'):
        rs2=dict(rs)
        for j in pooled:
            m=M[G,j]; p=P[m,j].copy(); s=S[G[m]]
            for sl in np.unique(s):
                k=s==sl
                if mode=='center': p[k]=p[k]-np.median(p[k])
                elif mode=='robustz': q1,md,q3=np.percentile(p[k],[25,50,75]); p[k]=(p[k]-md)/((q3-q1)/1.349)
                else: p[k]=z(p[k])
            rs2[names[j]]=R(p,Y[G[m],j])
        print(' ',mode,{n:round(rs2[n]-rs[n],4) for n in names if n in rs2 and n.startswith('grp')},'dir GOAL',round(np.mean(list(rs2.values())),4))
    # cross-output: grp label vs single outputs on 'all' test cells
    pairs={'grp:CD45':['single:CD45'],'grp:CD16':['single:CD16'],'grp:CD3/CD19/CD56':['single:CD3','single:CD19','single:CD56']}
    for g,ss in pairs.items():
        j=ix(g); m=M[G,j]; y=Y[G[m],j]
        own=P[m,j]; cr={s:round(R(P[m,ix(s)],y),3) for s in ss}
        sumz=np.sum([z(P[m,ix(s)]) for s in ss],0)
        avg=z(own)+z(sumz)
        print(' ',g,'own',round(R(own,y),3),'cross',cr,'sumz',round(R(sumz,y),3),'avg(own,cross)',round(R(avg,y),3))
    for s,g in [('single:CD45','grp:CD45'),('single:CD16','grp:CD16'),('single:CD3','grp:CD3/CD19/CD56'),('single:CD19','grp:CD3/CD19/CD56'),('single:CD56','grp:CD3/CD19/CD56'),('single:HLA-DR','grp:CD123/HLA-DR/CD14'),('single:CD14','grp:CD123/HLA-DR/CD14'),('single:CD123','grp:CD123/HLA-DR/CD14')]:
        j=ix(s); m=M[G,j]; y=Y[G[m],j]; own=P[m,j]; cr=P[m,ix(g)]
        print('  ',s,'own',round(R(own,y),3),'via',g,round(R(cr,y),3),'avg',round(R(z(own)+z(cr),y),3), 'avg .25',round(R(z(own)+.33*z(cr),y),3))
    # spearman per slide b01 grp
    for j in pooled:
        m=M[G,j]; p=P[m,j]; y=Y[G[m],j]; s=S[G[m]]
        for sl in np.unique(s):
            k=s==sl; print('   ',names[j],SN[sl][-18:],'pearson',round(R(p[k],y[k]),3),'spearman',round(spearmanr(p[k],y[k])[0],3),'n',k.sum())
