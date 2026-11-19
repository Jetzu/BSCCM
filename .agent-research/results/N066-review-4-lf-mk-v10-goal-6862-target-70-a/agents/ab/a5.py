import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
T=np.load(W+'/targets_v5cx_v10.npz',allow_pickle=True); Y,M,sp,b,h=T['Y'],T['mask'],T['split'],T['batch'],T['hist_row']; names=[str(n) for n in T['names']]
idx=pd.read_parquet(W+'/bsccm_index.parquet',columns=['antibodies','batch']); ab=idx.antibodies.values
G4=['CD123/HLA-DR/CD14','CD3/CD19/CD56','CD45','CD16']
sm=pd.read_parquet(W+'/bsccm_sm.parquet',columns=[f'{g}_full_model_unmixed' for g in G4]+[f'{m}_single_antibody_model_unmixed' for m in ['CD45','CD16','CD3','CD19','CD56','HLA-DR','CD14','CD123']])
CAL=(sp==0)&(h<0)
for g,singles in [('CD45',['CD45']),('CD16',['CD16']),('CD3/CD19/CD56',['CD3','CD19','CD56']),('CD123/HLA-DR/CD14',['CD123','HLA-DR','CD14'])]:
    x=sm[f'{g}_full_model_unmixed'].values; j=names.index('grp:'+g)
    for bb in (0,1):
        ref=(ab=='unstained')&(b==bb)&CAL; med=np.median(x[ref]); mad=np.median(np.abs(x[ref]-med))*1.4826
        al=(ab=='all')&(b==bb); t=np.arcsinh((x-med)/mad); mu,sd=t[al&CAL].mean(),t[al&CAL].std(); zz=(t-mu)/sd
        qa=np.percentile(zz[al],[5,25,50,75,95,99])
        line=f'grp:{g:18s} d{bb} unstMAD {mad:7.2f} all-slide q[5,25,50,75,95,99] {np.round(qa,2)}'
        print(line)
        for m in singles:
            s=(ab==m)&(b==bb)&(h<0); js=names.index('single:'+m)
            xs=sm[f'{m}_single_antibody_model_unmixed'].values
            r_raw=np.corrcoef(x[s],xs[s])[0,1]; r_lab=np.corrcoef(zz[s],Y[s,js])[0,1]
            print(f'    on {m:6s} slide n={s.sum():6d}: r(full,single raw) {r_raw:.3f}  r(grp-transformed full, single v10 label) {r_lab:.3f}  q {np.round(np.percentile(zz[s],[5,25,50,75,95,99]),2)}')
        # negative control: on unstained slide
        u=(ab=='unstained')&(b==bb)&(h<0); print(f'    unstained q {np.round(np.percentile(zz[u],[5,50,95,99]),2)}')
# C1: label-ratio vs direction asymmetry
ps=pd.read_csv('/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N066-review-4-lf-mk-v10-goal-6862-target-70-a/n066_perslide.csv'); ps=ps[ps.slide=='ALL']
d=ps.pivot(index='target',columns='direction',values='r'); d['asym']=d.b10-d.b01
tr0=(b==0)&(h<0); tr1=(b==1)&(h<0)
d['logratio']=[np.log(M[tr1,names.index(t)].sum()/M[tr0,names.index(t)].sum()) for t in d.index]
print(d.round(3)); print('corr(asym, log n1/n0)',round(np.corrcoef(d.asym,d.logratio)[0,1],3), 'singles only', round(np.corrcoef(d.asym[d.index.str.startswith('single')],d.logratio[d.index.str.startswith('single')])[0,1],3))
