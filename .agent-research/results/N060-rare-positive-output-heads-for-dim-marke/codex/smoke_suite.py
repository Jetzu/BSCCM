import os, subprocess, sys
from pathlib import Path
import numpy as np
import torch
root=Path('/tmp/codex_n060'); here=Path(__file__).resolve().parent; py=sys.executable
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', WANDB_MODE='disabled')
base=[py,str(here/'smoke_cpu.py'),'--root',str(root/'fixture')]
common=['--arch','resnet18','--epochs','1','--bs','8','--max-train','64','--workers','0','--targets','targets_v5cx_v10.npz','--seed','0']
arms={'hlgauss':['--head','hlgauss'],'twopart':['--head','twopart'],
      'fov_re':['--fov-re','.01','--w-autofl','0'],
      'pair':['--pair-loss','.5','--fov-batch','2','--w-autofl','0'],
      'autofl0':['--w-autofl','0'],'panel':['--panel-head']}
def run(name, protocol, args, script=None, cpu=True):
    out=root/'runs'/name
    cmd=base+(['--script',str(script)] if script else [])+['--','--out',str(out),'--protocol',protocol]+common+(['--device','cpu'] if cpu else [])+args
    print('COMMAND', ' '.join(cmd),flush=True)
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    print(proc.stdout,flush=True)
    if proc.returncode: raise RuntimeError(f'{name} failed: {proc.returncode}')
    s=torch.load(out/'ckpt.pt',map_location='cpu',weights_only=False)
    t=np.load(root/'fixture'/'targets_v5cx_v10.npz')
    assert len(np.unique(t['batch'][s['tr']]))==1
    assert not np.intersect1d(s['tr'],s['ev']).size
    pred=np.load(out/'eval_preds.npz')
    assert pred['P'].shape==(len(s['ev']),13) and np.isfinite(pred['P']).all()
    assert np.isfinite(s['log'][-1]['train_loss'])
    assert all(torch.isfinite(v).all() for v in s['m'].values() if v.dtype.is_floating_point)
    if 'readout.offset' in s['m']: assert s['m']['readout.offset'].abs().max() > 0
    if 'readout.pool_shift' in s['m']: assert s['m']['readout.pool_shift'].abs().max() > 0
    if 'reg.thresholds' in s['m']:
        for i, j in enumerate(s['m']['reg.indices'].tolist()):
            v = t['Y'][s['tr'], j][t['mask'][s['tr'], j]]
            assert np.isclose(s['m']['reg.thresholds'][i], np.quantile(v, .98))
    print('PASS',name,'finite loss/state/P, 13 targets, disjoint source/eval rows',flush=True)
    return s
for protocol in ['b1to0','b0to1']:
    for name,args in arms.items():
        tag=name+'_'+protocol
        first=run(tag,protocol,args)
        if name in ('hlgauss','twopart','fov_re','panel'):
            second=run(tag,protocol,args)
            assert all(torch.equal(v,second['m'][k]) for k,v in first['m'].items())
            print('PASS',tag,'finished-checkpoint resume preserves state',flush=True)
# Default parity: adapt only device placement, loader CPU requirements and
# autocast in the ORIGINAL harness; leave its model/loss/eval/config untouched.
original=Path('/home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py').read_text()
old=original.replace('.cuda(non_blocking=True)', ".to('cpu',non_blocking=True)").replace('.cuda()', ".to('cpu')")
old=old.replace("device='cuda'", "device='cpu'").replace("GPUNorm('cuda')", "GPUNorm('cpu')").replace("map_location='cuda'", "map_location='cpu'")
old=old.replace("with torch.autocast('cuda', torch.bfloat16):", "with torch.autocast('cpu',torch.bfloat16,enabled=False):")
old=old.replace('pin_memory=True, persistent_workers=True, prefetch_factor=8','pin_memory=False')
legacy=root/'legacy_cpu.py'; legacy.write_text(old)
a=run('legacy','b0to1',[],legacy,False)
b=run('new_default','b0to1',[])
assert all(torch.equal(v,b['m'][k]) for k,v in a['m'].items())
pa=np.load(root/'runs/legacy/eval_preds.npz')['P']; pb=np.load(root/'runs/new_default/eval_preds.npz')['P']
assert np.array_equal(pa,pb)
b['cfg'].pop('device'); assert a['cfg']==b['cfg']
print('PASS default model tensors, loss, eval P byte-equal to CPU-adapted original; cfg equal except explicit device cpu',flush=True)
# Exercise a legacy checkpoint with the new default config: CPU-placement shim
# preserves the actual parser defaults and _cfg(), without a --device flag.
shim=root/'new_cpu_default.py'; shim.write_text((here/'train_lf.py').read_text().replace('DEVICE = a.device', "DEVICE = 'cpu'"))
c=run('legacy','b0to1',[],shim,False)
assert all(torch.equal(v,c['m'][k]) for k,v in a['m'].items())
print('PASS original checkpoint resumes under new default config and state layout (CPU placement shim)',flush=True)
print('ALL 12 ARM/DIRECTION SMOKES + RESUMES + DEFAULT PARITY PASSED',flush=True)
