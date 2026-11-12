import numpy as np, pandas as pd
W='/hnvme/workspace/v181bb10-bsccm'
sm=pd.read_parquet(W+'/bsccm_sm.parquet'); ix=pd.read_parquet(W+'/bsccm_index.parquet')
ch='500-550'
cols=[c for c in sm.columns if ch in c]; print(sm.loc[ix.antibodies=='CD19', cols].describe().T)
s=ix.antibodies.values
for m in ['CD19','unstained','CD56','all']:
    sel=s==m; print(m, {c: float(np.isfinite(sm.loc[sel,c]).mean()) for c in ['CD19_single_antibody_model_unmixed','autofluor_single_antibody_model_unmixed','CD3/CD19/CD56_full_model_unmixed']})
sel=(s=='CD19')
x=sm.loc[sel]; print(np.corrcoef(x[f'Fluor_{ch}_shading_corrected'], x[f'Fluor_{ch}_total_raw']-x[f'Fluor_{ch}_background'])[0,1])
print(np.corrcoef(x[f'Fluor_{ch}_shading_corrected'], (x[f'Fluor_{ch}_total_raw']-x[f'Fluor_{ch}_background_lowess'])/x[f'Fluor_{ch}_shading'])[0,1])
print(ix.detection_radius.describe())
print(x[['CD19_single_antibody_model_unmixed','autofluor_single_antibody_model_unmixed']].describe())
