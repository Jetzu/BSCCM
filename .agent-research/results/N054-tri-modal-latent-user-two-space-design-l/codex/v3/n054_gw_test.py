#!/usr/bin/env -S python3 -I -B
"""CPU numerical unit test: GW invariance, point recovery, balanced marginals."""
import importlib.util
import json
from pathlib import Path
import torch
from torch.nn import functional as F
spec=importlib.util.spec_from_file_location('n054_gw',Path(__file__).with_name('n054_trimodal.py'))
n=importlib.util.module_from_spec(spec); spec.loader.exec_module(n)

def test():
    torch.set_num_threads(2); records=[]
    for seed,count,dim in [(0,48,16),(1,48,16),(2,48,16),(3,64,256)]:
        torch.manual_seed(seed)
        x=F.normalize(torch.randn(count,dim,dtype=torch.float64),dim=-1)
        q,_=torch.linalg.qr(torch.randn(dim,dim,dtype=torch.float64)); perm=torch.randperm(len(x))
        y=x[perm]@q; coupling=n.entropic_gw(x,y); reference=n.entropic_gw(x,x[perm])
        recovered=perm[coupling.argmax(1)]
        identity=float((recovered==torch.arange(len(x))).double().mean())
        row_error=float((coupling.sum(1)-1/len(x)).abs().max()); col_error=float((coupling.sum(0)-1/len(x)).abs().max())
        rotation_error=float((coupling-reference).abs().max())
        assert identity==1.0 and max(row_error,col_error)<1e-8 and rotation_error<1e-8
        assert torch.isfinite(coupling).all() and coupling.min()>=0
        records.append(dict(seed=seed,dim=dim,n=len(x),identity_rate=identity,chance=1/len(x),row_error=row_error,col_error=col_error,rotation_error=rotation_error))
    return dict(status='passed',cases=records)

if __name__=='__main__':
    result=test(); print(json.dumps(result,indent=2))
    Path(__file__).with_name('gw_test_results.json').write_text(json.dumps(result,indent=2)+'\n')
