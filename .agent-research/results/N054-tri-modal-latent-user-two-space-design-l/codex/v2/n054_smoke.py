#!/usr/bin/env python3
"""Real-data, CPU-only integration and signal/resume audit of all three units."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scratch',default=f'/tmp/codex_n054/v2_smoke_{time.time_ns()}')
    a=ap.parse_args(); scratch=Path(a.scratch).resolve(); assert scratch.is_relative_to('/tmp/codex_n054')
    scratch.mkdir(parents=True,exist_ok=True); here=Path(__file__).resolve().parent
    env=os.environ.copy(); env.update(TMPDIR='/tmp/codex_n054',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    begin=time.time(); records=[]
    def call(cmd,interrupt=False,expected=0):
        print('COMMAND',' '.join(map(str,cmd)),flush=True)
        proc=subprocess.Popen(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
        sent=False; sent_at=None
        for line in proc.stdout:
            print(line,end='',flush=True)
            if interrupt and not sent and '"job": "gs_encoder"' in line and '"step": 10,' in line:
                proc.send_signal(signal.SIGTERM); sent=True; sent_at=time.time()
        code=proc.wait(); expected=143 if interrupt else expected
        assert code==expected,(code,expected,cmd)
        if interrupt: assert sent,'did not interrupt live stage'
        elapsed=time.time()-sent_at if sent_at else None
        records.append(dict(command=list(map(str,cmd)),exit=code,signal_to_exit_seconds=elapsed))
        return elapsed
    base=[sys.executable,'-I',here/'n054_trimodal.py']
    flags=['--stage','all','--smoke','--device','cpu','--threads','2']
    lf=scratch/'n054v2_lf'; b1=scratch/'n054v2_batch1'; b0=scratch/'n054v2_batch0'
    lf_cmd=base+flags+['--unit','lf','--out',lf]
    call(lf_cmd)
    b1_cmd=base+flags+['--unit','batch1','--out',b1,'--lf-unit',lf]
    latency=call(b1_cmd,interrupt=True)
    import numpy as np
    import torch
    ck=torch.load(b1/'space2/ckpt.pt',map_location='cpu',weights_only=False)
    interrupted_step=ck['jobs']['gs_encoder']['step']; assert 10<=interrupted_step<20
    assert not (b1/'space2/gs_encoder.pt').exists()
    lf_before={str(f):f.stat().st_mtime_ns for f in lf.rglob('*.pt')}
    call(b1_cmd)
    call(base+flags+['--unit','batch0','--out',b0,'--lf-unit',lf])
    # Entire completed units must skip every stage and preserve artifact mtimes.
    all_before={str(f):f.stat().st_mtime_ns for f in scratch.rglob('*.pt')}
    call(lf_cmd); call(b1_cmd)
    assert all_before=={str(f):f.stat().st_mtime_ns for f in scratch.rglob('*.pt')}
    assert lf_before=={str(f):f.stat().st_mtime_ns for f in lf.rglob('*.pt')}
    # Same production training path, uninterrupted, with identical initial seed.
    reference=scratch/'resume_reference'
    call(base+['--stage','gs_encoder','--unit','batch1','--holdout','batch1','--out',reference,
        '--space-dir',lf/'space1','--ridge-cache',lf/'ridge.npz','--lf-unit',lf,
        '--arm','A1-LF','--smoke','--device','cpu','--threads','2'])
    interrupted=torch.load(b1/'space2/gs_encoder.pt',map_location='cpu',weights_only=False)
    uninterrupted=torch.load(reference/'gs_encoder.pt',map_location='cpu',weights_only=False)
    assert all(torch.equal(v,uninterrupted['state'][k]) for k,v in interrupted['state'].items()),'resume weights differ'
    code=hashlib.sha256((here/'n054_trimodal.py').read_bytes()).hexdigest()
    targets=np.load('/hnvme/workspace/v181bb10-bsccm/targets_tt6c.npz')
    capacities=set(); run_summaries=[]; shared_sources=[]; nclinical=0
    for unit in (b1,b0):
        sel=json.loads((unit/'selection.json').read_text()); assert sel['code_signature']==code
        chosen={r['run'] for r in sel['selected'].values()}
        assert len(list(unit.rglob('gs_encoder.pt')))==3,'duplicated GS fits'
        pairs=np.load(unit/'space2/source_cache/pairs.npz')
        for split,key in [(0,'train'),(1,'val')]:
            rows=pairs[key+'_rows']; assert (targets['hist_row'][rows]>=0).all()
            assert (targets['split'][rows]==split).all() and (targets['batch'][rows]==1-int(unit.name[-1])).all()
        for flow in unit.glob('*/flow.pt'):
            f=torch.load(flow,map_location='cpu',weights_only=False); args=f['args']; capacities.add(f['capacity'])
            assert Path(args['space_dir'])==lf/'space1' and Path(args['gs_dir'])==unit/'space2'
            shared_sources.append(dict(run=str(flow.parent),space=args['space_dir'],gs=args['gs_dir']))
            for rec in f['endpoint_audit']:
                assert rec['same_cell_assignments']==0 and rec['target_gs']==0 and rec['source_gs']==rec['n']
            rows=f['source_rows']; assert (targets['split'][rows]==0).all()
            if args['arm']=='A1-LF': assert (targets['hist_row'][rows]<0).all()
            else: assert (targets['hist_row'][rows]>=0).all() and (targets['batch'][rows]==1-int(unit.name[-1])).all()
            if args['arm']=='A1-GS': assert args['init_flow']==str(unit/f"A1-LF_s{args['sigma_bridge']:g}"/'flow.pt')
        for output in unit.glob('*/eval.json'):
            result=json.loads(output.read_text()); run=output.parent
            assert result['M3']['n']==150
            assert result['audit']['gs_in_lf_training']==0 and result['audit']['test_slide_in_gs_training']==0
            assert result['audit']['n_lf_train']+result['audit']['n_lf_val']+result['audit']['n_lf_test']<=2000
            expected=result['arm']=='A0' or str(run.resolve()) in chosen
            assert (run/'clinical/clinical.json').exists()==expected,'clinical ran on an unselected config'
            if expected:
                clinical=json.loads((run/'clinical/clinical.json').read_text()); assert len(clinical['contrasts'])==25; nclinical+=1
            run_summaries.append(dict(unit=unit.name,arm=result['arm'],M1=result['M1_autofl_truth_r'],selected=expected))
    assert len(capacities)==1,'direct/flow capacities differ'
    completion=lf/'completion/completion'; report=json.loads((completion/'completion.json').read_text())
    assert report['audit']['raw_single_autofl_pairs']==312590 and report['audit']['raw_failed_group_draw2']==0
    assert all(r['scored_visible']==0 and r['validation_scored']<r['validation_observed'] for r in report['validation_masks'])
    assert len([r for r in report['metrics'] if r['test']=='C1' and r['observed_marker']!='all'])==24
    for file in completion.glob('*_C[12]_*.npz'):
        if file.stem.endswith('_ridge'): continue
        z=np.load(file); j=int(file.stem.rsplit('_',1)[1]); assert not z['input_mask'][:,j].any()
        assert np.isfinite(z['lf']).all() and np.isfinite(z['fused']).all() and np.isfinite(z['labels_mlp']).all()
    for protocol in ('fov','draw0_to_1','draw1_to_0'):
        ck=torch.load(completion/(protocol+'.pt'),map_location='cpu',weights_only=False)
        assert (targets['hist_row'][ck['train_rows']]<0).all() and (targets['split'][ck['train_rows']]==0).all()
        if protocol.startswith('draw'):
            assert (targets['batch'][ck['train_rows']]==int(protocol[4])).all()
            assert (targets['batch'][ck['test_rows']]==1-int(protocol[4])).all()
    tables=scratch/'n054v2_tables'
    call([sys.executable,'-I',here/'n054_aggregate.py','--units',b1,b0,'--lf-unit',lf,'--out',tables])
    agg=json.loads((tables/'aggregation.json').read_text()); assert agg['n_primary']==10 and agg['n_grid']==24 and agg['n_diagnostics']==6
    # Code signature must reject a stale completed unit, even with identical args.
    changed=scratch/'changed.py'; changed.write_bytes((here/'n054_trimodal.py').read_bytes()+b'\n# changed signature\n')
    call([sys.executable,'-I',changed]+flags+['--unit','lf','--out',lf],expected=1)
    # Numerical regression guard for the dimension-averaging bug and AF duplicates.
    spec=importlib.util.spec_from_file_location('n054_test',here/'n054_trimodal.py'); mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    y=torch.zeros(3,13); y[:,12]=torch.tensor([0.,.1,.5]); m=torch.zeros(3,13,dtype=torch.bool); m[:,12]=True
    dup=mod.near_duplicate(y,m); assert dup[0,1] and not dup[0,2] and not dup.diagonal().any()
    zg=torch.randn(4,256,requires_grad=True); zl=torch.randn_like(zg)
    summed=(zg-zl).square().sum(-1).mean(); averaged=(zg-zl).square().mean()
    gsum,=torch.autograd.grad(summed,zg,retain_graph=True); gmean,=torch.autograd.grad(averaged,zg)
    assert torch.allclose(gsum,256*gmean)
    summary=dict(status='passed',seconds=round(time.time()-begin,2),scratch=str(scratch),code_signature=code,
        units=['lf','batch1','batch0'],signal='SIGTERM via --stage all driver',interrupted_step=interrupted_step,
        signal_to_exit_seconds=latency,resume_bitwise_equal=True,completed_unit_mtimes_unchanged=True,
        gs_fits=6,lf_space_fits=2,completion_protocol_fits=3,clinical_evaluations=nclinical,
        clinical_images_per_run=dict(PBC=16,MLL23=24),primary_rows=agg['n_primary'],grid_rows=agg['n_grid'],
        completion_rows=len(report['metrics']),per_observed_marker_C1_rows=24,direct_and_flow_capacity=capacities.pop(),
        shared_sources=shared_sources,runs=run_summaries,commands=records)
    (here/'smoke_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    import shutil
    artifacts=here/'smoke_artifacts'; exported=[]
    for f in sorted(scratch.rglob('*')):
        if f.is_file() and f.suffix in ('.csv','.json') and 'resume_reference' not in str(f):
            dest=artifacts/f.relative_to(scratch); dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(f,dest); exported.append(str(dest.relative_to(here)))
    (artifacts/'index.json').write_text(json.dumps(dict(source=str(scratch),files=exported),indent=2)+'\n')
    print('SMOKE PASSED',json.dumps({k:v for k,v in summary.items() if k not in ('commands','runs','shared_sources')}),flush=True)

if __name__=='__main__': main()
