#!/usr/bin/env -S python3 -I -B
"""PBC cross-half hygiene audit. No clinical predictions or model selection."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent

def names_in(z):
    return sorted(n for n in z.namelist() if n.lower().endswith('.jpg') and '__MACOSX' not in n and not Path(n).name.startswith('.'))

def fingerprint(z):
    # Strong binding to archive directory; avoids rereading multi-GB archive per run.
    s='\n'.join(f'{i.filename}\t{i.CRC}\t{i.file_size}' for i in z.infolist())
    return hashlib.sha256(s.encode()).hexdigest()

def excluded(path, audit=None):
    audit=Path(audit or HERE/'pbc_excluded.json')
    obj=json.loads(audit.read_text())
    with zipfile.ZipFile(path) as z:
        if fingerprint(z)!=obj['archive_directory_sha256']:raise ValueError('PBC archive differs from duplicate audit')
    return set(obj['excluded'])

def audit(path,out):
    import torch
    torch.set_num_threads(12)
    with zipfile.ZipFile(path) as z:
        names=names_in(z);A=set(names[::2]);B=set(names[1::2]);groups=defaultdict(list)
        for n in names:
            i=z.getinfo(n);groups[(i.CRC,i.file_size)].append(n)
        exact=[dict(crc32=f'{crc:08x}',size=size,names=ns) for (crc,size),ns in groups.items() if set(ns)&A and set(ns)&B]
        thumbs=[]
        for i,n in enumerate(names):
            im=Image.open(io.BytesIO(z.read(n))).convert('RGB').resize((24,24),Image.Resampling.BILINEAR)
            thumbs.append(np.asarray(im,np.float32).ravel()/255.)
            if (i+1)%4000==0:print('thumbnail audit',i+1,flush=True)
        x=torch.from_numpy(np.array(thumbs));x-=x.mean(1,keepdim=True);x/=x.norm(dim=1,keepdim=True)+1e-8
        near=[]
        for s in range(0,len(names[1::2]),256):
            sim=x[1::2][s:s+256]@x[::2].T
            for bi,ai in (sim>=.995).nonzero().tolist():
                na,nb=names[2*ai],names[2*(s+bi)+1]
                near.append(dict(a=na,b=nb,cosine=float(sim[bi,ai]),
                                 a_crc32=f'{z.getinfo(na).CRC:08x}',a_size=z.getinfo(na).file_size,
                                 b_crc32=f'{z.getinfo(nb).CRC:08x}',b_size=z.getinfo(nb).file_size))
        drop=set(n for g in exact for n in g['names'])|set(n for g in near for n in [g['a'],g['b']])
        obj=dict(archive=str(path),archive_directory_sha256=fingerprint(z),threshold=.995,
                 thumbnail='RGB24x24 bilinear; row centered; L2 cosine',exact=exact,near=near,excluded=sorted(drop),
                 original_counts=dict(A=len(A),B=len(B)),remaining_counts=dict(A=len(A-drop),B=len(B-drop)))
    Path(out).write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({k:obj[k] for k in ['original_counts','remaining_counts']}), 'excluded',len(drop),'exact groups',len(exact),'near pairs',len(near),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--zip',default='/hnvme/workspace/v181bb10-bsccm/external/pbc/PBC_dataset_normal_DIB.zip');p.add_argument('--out',default=str(HERE/'pbc_excluded.json'))
    a=p.parse_args();audit(a.zip,a.out)
if __name__=='__main__':main()
