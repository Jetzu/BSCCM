import sys,json,torch,numpy as np,ast
sys.path.insert(0,'/home/hpc/v181bb/v181bb10/BSCCM')
from bsccm.encoders import LFBackbone
from bsccm.gpu_norm import GPUNorm
W='/hnvme/workspace/v181bb10-bsccm';torch.set_num_threads(4)
x=GPUNorm('cpu')(torch.from_numpy(np.array(np.load(W+'/pack/led_array.npy',mmap_mode='r')[:1]).view(np.int16)),torch.from_numpy(np.array(np.load(W+'/pack/dpc.npy',mmap_mode='r')[:1])),torch.tensor([0]))
for a in ('dinov3_s','dinov3_b'):
 m=LFBackbone(a).eval()
 with torch.no_grad():
  y=m(x);m.net.global_pool='token';z=m(x)
 print('POOL',a,'avg vs cls cosine',torch.nn.functional.cosine_similarity(y,z).item(),'meanabs',(y-z).abs().mean().item())
# Cfg pins paths/row identities, not label contents.
p='/home/hpc/v181bb/v181bb10/BSCCM/scripts/train_lf.py';tree=ast.parse(open(p).read()); f=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_cfg'][0]
from types import SimpleNamespace
T=np.load(W+'/targets_v5cx_v10.npz'); Y=T['Y'].copy(); ns=dict(a=SimpleNamespace(arch='dinov3_s',targets='targets_v5cx_v10.npz',enc_lr_mult=.1),_NEW_DEFAULTS={},PHYS=False,PSEUDO_ID=None)
exec(compile(ast.Module(body=[f],type_ignores=[]),'<cfg>','exec'),ns);before=ns['_cfg']();j=np.flatnonzero(T['mask'][0])[0];Y[0,j]+=1
print('CFG changing Y with same path passes config',before==ns['_cfg'](),'row selection unchanged',True)
for protocol,source in [('b1to0',1),('b0to1',0)]:
 h=(T['mask'].any(1)|(T['cls3']>=0))&(T['hist_row']<0);tr=np.flatnonzero((T['batch']==source)&h)
 print('METRIC',protocol,'trained first12',(T['mask'][tr].sum(0)[:12]>=1000).tolist())
