from pathlib import Path
import ast,json
import numpy as np
R=Path('/home/hpc/v181bb/v181bb10/BSCCM'); W=Path('/hnvme/workspace/v181bb10-bsccm')
source=ast.parse((R/'scripts/rescore_targets.py').read_text())
proto_assign=next(n for n in ast.walk(source) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='proto' for t in n.targets))
expr=compile(ast.Expression(proto_assign.value),'<rescore protocol>','eval')
for run in ['n014_sslinit_b1to0','n014_sslinit_b0to1','n014_sslinit_b1to0_renamed','model_copy','n023_v5_b01','n023_v5_b01_note_b100']:
    print('PROTOCOL',run,eval(expr,{'run':run}))
A=np.load(W/'targets_v4.npz'); B=np.load(W/'targets_v5.npz'); b,sp,hr,sl=A['batch'],A['split'],A['hist_row'],A['slide']; y,v=A['Y'],B['Y']; mask=B['mask']; names=list(A['names'])
for proto in ('fov','b1to0','b0to1'):
    tr=sp==0 if proto=='fov' else b==(1 if proto=='b1to0' else 0)
    tr &= hr<0
    tr=np.flatnonzero(tr)
    # Supported max-train option; source evaluator counts actual rows, rescorer does not.
    tr=np.random.default_rng(0).permutation(tr)[:1000]
    expected=mask[tr].sum(0)>=1000
    trb={'b1to0':1,'b0to1':0}.get(proto)
    got=mask[b==trb].sum(0)>=1000 if trb is not None else np.ones(len(names),bool)
    print('COUNT MISMATCH max_train=1000',proto,'actual train counts',mask[tr,:12].sum(0).tolist(),'actual trained markers',int(expected[:12].sum()),'rescore trained markers',int(got[:12].sum()))
D=json.load(open('audit_rescore.json')); d1,d2=D[:2]
old=(d1['v4']['all']+d2['v4']['all'])/2; common=(d1['v4']['common11']+d2['v4']['common11'])/2; new=(d1['v5']['valid']+d2['v5']['valid'])/2
print('GOAL DECOMPOSITION','v4_12',old,'v4_common11',common,'v5_common11',new,'drop_target_gain',common-old,'normalization_gain',new-common,'fraction_gain_from_dropping',(common-old)/(new-old))
# Show per-slide affine invariance for the exact GS evaluation cohorts and label calibration overlap.
for bb in (0,1):
    r=(b==bb)&(hr>=0)&mask[:,-1]
    slope,intercept=np.polyfit(y[r,-1],v[r,-1],1)
    print('GS AFFINE',bb,'n_eval',int(r.sum()),'n_calibration_overlap',int((r&(sp==0)).sum()),'slope',slope,'intercept',intercept,'max_abs_affine_error',np.max(np.abs(v[r,-1]-(slope*y[r,-1]+intercept))),'corr',np.corrcoef(y[r,-1],v[r,-1])[0,1])
