import numpy as np
W='/hnvme/workspace/v181bb10-bsccm'; T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); S=T['slide']; SN=np.array([s.split('/')[0].replace('antibodies_','')+'/'+s.split('/')[-1][-1] for s in T['slide_names'].astype(str)])
names=[str(n) for n in T['names']]
pooled=[names.index(n) for n in ('grp:CD3/CD19/CD56','grp:CD45','grp:CD16')]
L=lambda r: np.load(f'{W}/runs/{r}/eval_preds.npz')
def comp(tp,pa,d):
    a,bq=L(f'{tp}_{d}'),L(f'{pa}_{d}'); P=a['P'].copy(); P[:,pooled]=bq['P'][:,pooled]; return P,a['G']
for d in ('b10','b01'):
    base=np.mean([comp(f'n063_tpfov_s{s}',f'n063_panel_s{s}',d)[0] for s in (2,3)],0)
    z2,G=comp('n065_tpfov_z2','n065_panel_z2',d); cy,_=comp('n065_tpfov_cyto','n065_panel_cyto',d)
    P=np.mean([base,z2,cy]+[L(f'{r}_{d}')['P'] for r in ('n048_v9_base','n048_v9_zoom2','n053_cytoland')],0)
    s=S[G]; us=np.unique(s)
    print(d,'pred mean per test slide (z vs all held-out cells), own-target slide marked *')
    hdr='target'.ljust(20)+''.join(SN[u][:11].rjust(12) for u in us); print(hdr)
    for j in list(range(8))+pooled:
        p=P[:,j]; mu,sd=p.mean(),p.std(); own=names[j].split(':')[1]
        cells=[]
        for u in us:
            k=s==u; v=(p[k].mean()-mu)/sd; mark='*' if SN[u].split('/')[0]==own or (names[j].startswith('grp') and SN[u].startswith('all')) else ' '
            cells.append(f'{v:+.2f}{mark}'.rjust(12))
        print(names[j][:20].ljust(20)+''.join(cells))
    print('n per slide',[int((s==u).sum()) for u in us])
