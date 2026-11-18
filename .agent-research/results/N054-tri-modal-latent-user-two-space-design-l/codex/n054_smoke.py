#!/usr/bin/env python3
"""Real-data CPU integration smoke; all scratch stays under /tmp/codex_n054.
Runs both directions, both sigma settings, both hub SigLIP settings, baseline,
completion, clinical inference, a SIGUSR1 restart, and resume equivalence.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--scratch',default=f'/tmp/codex_n054/smoke_{time.time_ns()}')
    ap.add_argument('--data',default='/hnvme/workspace/v181bb10-bsccm'); args=ap.parse_args()
    scratch=Path(args.scratch).resolve(); assert scratch.is_relative_to('/tmp/codex_n054')
    scratch.mkdir(parents=True,exist_ok=True); here=Path(__file__).resolve().parent
    env=os.environ.copy(); env.update(TMPDIR='/tmp/codex_n054',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    records=[]; started=time.time()
    def call(cmd,signal_at_ten=False):
        print('COMMAND', ' '.join(map(str,cmd)),flush=True)
        p=subprocess.Popen(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,env=env)
        sent=False
        for line in p.stdout:
            print(line,end='',flush=True)
            if signal_at_ten and not sent and '"step": 10,' in line:
                p.send_signal(signal.SIGUSR1); sent=True
        code=p.wait()
        if signal_at_ten:
            assert sent and code==99,('signal test failed',sent,code)
        else: assert code==0,('command failed',code)
        return code
    variants=[('A1',0,0),('A1',.1,0),('A2',0,0),('A2',.1,0),('A3',0,0),('A3',0,.5)]
    for holdout in ('batch1','batch0'):
        for arm,sigma,w in variants:
            label=f'{arm}_s{sigma:g}_w{w:g}_{holdout}'; out=scratch/label
            base=['--out',out,'--arm',arm,'--holdout',holdout,'--sigma-bridge',str(sigma),'--w-sig',str(w),
                  '--data',args.data,'--smoke','--threads','2','--ridge-cache',scratch/'ridge.npz']
            cmd=[sys.executable,'-I',here/'n054_trimodal.py']
            call(cmd+['lf_spaces']+base)
            # Interrupt precisely one live GS training job, then resume.
            interrupt=arm=='A3' and w==0 and holdout=='batch1'
            if interrupt and not (out/'gs_encoder.pt').exists(): call(cmd+['gs_encoder']+base,True)
            call(cmd+['gs_encoder']+base)
            call(cmd+['flow']+base)
            call(cmd+['eval']+base+['--validation-only'])
            call(cmd+['eval']+base)
            result=json.loads((out/'eval.json').read_text()); assert result['M3']['n']==150
            assert result['audit']['n_lf_train']+result['audit']['n_lf_val']+result['audit']['n_lf_test']<=2000
            assert result['audit']['n_gs_train']+result['audit']['n_gs_val']+result['audit']['n_gs_test']<=300
            assert len(result['per_target'])==132  # 26 M2 + 26 same-draw M5 + 78 full/per-draw M5 + autofl check.
            records.append(dict(run=label,M1=result['M1_autofl_truth_r'],retrieval=result['M3'],cka=result['M4_GS_LF']['linear_cka']))
            if arm=='A3' and w==0 and holdout=='batch1':
                call(cmd+['completion']+base)
                # Completed-stage resume must be idempotent.
                call(cmd+['gs_encoder']+base)
        out=scratch/f'A0_{holdout}'
        call([sys.executable,'-I',here/'n054_trimodal.py','eval','--out',out,'--arm','A0','--holdout',holdout,
              '--data',args.data,'--smoke','--threads','2','--ridge-cache',scratch/'ridge.npz'])
        result=json.loads((out/'eval.json').read_text()); records.append(dict(run=f'A0_{holdout}',M1=result['M1_autofl_truth_r'],retrieval=result['M3']))
    # Clinical source labels are used only to calculate registered AUROCs.
    for arm,label in [('A0','A0_batch1'),('A1','A1_s0_w0_batch1'),('A2','A2_s0.1_w0_batch1'),('A3','A3_s0_w0.5_batch1')]:
        call([sys.executable,'-I',here/'n054_clinical.py','--run',scratch/label,'--out',scratch/(label+'_clinical'),
              '--arm',arm,'--data',args.data,'--smoke','--threads','2','--batch-size','4','--tta','0' if arm=='A0' else '1'])
        clinical=json.loads((scratch/(label+'_clinical')/'clinical.json').read_text())
        assert len(clinical['contrasts'])==25
    # Validate actual saved masks, rows and finite arrays rather than just exit codes.
    import numpy as np
    import torch
    c=scratch/'A3_s0_w0_batch1/completion'
    comp=json.loads((c/'completion.json').read_text())
    assert all(r['n']==0 for r in comp['metrics'] if r['test']=='C1')
    for file in c.glob('*_C2_*.npz'):
        z=np.load(file); j=int(file.stem.rsplit('_',1)[1]); assert not z['input_mask'][:,j].any()
        assert np.isfinite(z['lf']).all() and np.isfinite(z['fused']).all()
    for protocol in ('fov','draw0_to_1','draw1_to_0'):
        ck=torch.load(c/(protocol+'.pt'),map_location='cpu',weights_only=False)
        targets=np.load(Path(args.data)/'targets_tt6c.npz')
        assert (targets['hist_row'][ck['train_rows']]<0).all()
        assert (targets['split'][ck['train_rows']]==0).all()
        if protocol.startswith('draw'):
            source=int(protocol[4]); assert (targets['batch'][ck['train_rows']]==source).all()
            assert (targets['batch'][ck['test_rows']]==1-source).all()
    # Uninterrupted GS reference: same seed, LF model, data, optimizer schedule.
    import shutil
    reference=scratch/'resume_reference'; reference.mkdir(exist_ok=True)
    shutil.copy2(scratch/'A3_s0_w0_batch1/lf_spaces.pt',reference/'lf_spaces.pt')
    call([sys.executable,'-I',here/'n054_trimodal.py','gs_encoder','--out',reference,'--arm','A3','--holdout','batch1',
          '--data',args.data,'--smoke','--threads','2','--ridge-cache',scratch/'ridge.npz'])
    interrupted=torch.load(scratch/'A3_s0_w0_batch1/gs_encoder.pt',map_location='cpu',weights_only=False)
    uninterrupted=torch.load(reference/'gs_encoder.pt',map_location='cpu',weights_only=False)
    for k,v in interrupted['state'].items(): assert torch.equal(v,uninterrupted['state'][k]),('resume differs',k)
    summary=dict(status='passed',seconds=round(time.time()-started,2),runs=records,completion=comp['metrics'],
                 signal_restart='SIGUSR1 exit 99 then resume; bitwise equal to uninterrupted CPU GS training',
                 clinical='4 arms, 16 PBC images and 24 MLL23 images per arm; 25 contrasts each',scratch=str(scratch))
    artifact_root=here/'smoke_artifacts'; exported=[]
    for file in sorted(scratch.rglob('*')):
        if file.is_file() and file.suffix in ('.json','.csv','.npz'):
            target=artifact_root/file.relative_to(scratch); target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,target); exported.append(str(target.relative_to(here)))
    (artifact_root/'index.json').write_text(json.dumps(dict(source=str(scratch),files=exported),indent=2)+'\n')
    summary['artifacts']='smoke_artifacts/index.json'
    (here/'smoke_results.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('SMOKE PASSED',json.dumps({k:v for k,v in summary.items() if k not in ('runs','completion')}),flush=True)


if __name__=='__main__': main()
