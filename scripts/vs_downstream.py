"""N028 downstream check: does a Giemsa student read the same biology from VIRTUAL Wright images as from the REAL ones of the same cells?
Inputs per held-out cell (vs_eval generated.npz rows): (gen) generated 128 px virtual stain, (reg) the identically registered real GS crop,
(full) the real full-field GS crop the student was trained on. Student = a train_gs_bind checkpoint (ViT + PG + S heads), 8-view TTA.
Reports per input: within-slide Pearson of the student's autofl output vs measured autofl (same cells), and student agreement
r(gen, reg) / r(gen, full) per output. Usage: vs_downstream.py --ckpt gs_bind.pt --generated generated.npz --registration register.npz --out DIR"""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, timm
from vs_common import registered_rgb
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--generated', required=True); ap.add_argument('--registration', required=True)
ap.add_argument('--targets', nargs='+', default=['targets_v4c.npz', 'targets_v4.npz']); ap.add_argument('--dim', type=int, default=256); ap.add_argument('--out', required=True)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
W = '/hnvme/workspace/v181bb10-bsccm'
H = np.load(W + '/pack/histology.npy', mmap_mode='r'); scale_student = float(np.percentile(H[:500:5], 99.9))   # == train_gs_bind.py normalisation
names = [str(n) for n in np.load(W + '/' + a.targets[0], allow_pickle=True)['names']]; K = len(names)
mlp = lambda i, o: nn.Sequential(nn.Linear(i, 1024), nn.GELU(), nn.LayerNorm(1024), nn.Linear(1024, o))
vit = timm.create_model('vit_base_patch14_dinov2', pretrained=False, num_classes=0, img_size=224)
net = nn.ModuleDict({'vit': vit, 'PG': mlp(vit.num_features, a.dim), 'S': mlp(a.dim, K)})
sd = torch.load(a.ckpt, map_location='cpu', weights_only=True); net.load_state_dict({k: v for k, v in sd.items() if k.split('.')[0] in ('vit', 'PG', 'S')}, strict=True)
net = net.to(dev).eval(); MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
@torch.no_grad()
def predict(imgs):   # list of HxWx3 float arrays in the student's [0,1] units -> (N, K), mean over rot90 x flip
    out = []
    for i in range(0, len(imgs), 64):
        x = torch.stack([F.interpolate(torch.tensor(np.ascontiguousarray(b), dtype=torch.float32).permute(2, 0, 1)[None], size=224, mode='bilinear', antialias=True)[0] for b in imgs[i:i + 64]]).to(dev)
        x = (x.clamp(0, 1) - MEAN) / STD; P = 0
        for k in range(4):
            for f in (0, 1):
                v = torch.rot90(x, k, (2, 3)); v = torch.flip(v, (3,)) if f else v
                with torch.autocast(dev, torch.bfloat16, enabled=dev == 'cuda'):
                    z = F.normalize(net['PG'](net['vit'](v).float()), dim=-1); P = P + net['S'](z * np.sqrt(a.dim)).float()
        out.append((P / 8).cpu().numpy())
    return np.concatenate(out)
G = np.load(a.generated); R = np.load(a.registration); gidx, hrow, rgb_scale = G['gidx'], G['hist_row'], float(G['rgb_scale'])
gen = [G['images'][i].astype(np.float32) / 255. * rgb_scale / scale_student for i in range(len(hrow))]   # red-team m3: student units
reg = [registered_rgb(np.asarray(H[r], np.float32), R['scale'][r], R['dy'][r], R['dx'][r], 128) / scale_student for r in hrow]
full = [np.asarray(H[r], np.float32) / scale_student for r in hrow]
P = {k: predict(v) for k, v in (('gen', gen), ('reg', reg), ('full', full))}
pear = lambda x, y: float(np.corrcoef(x, y)[0, 1])
res = dict(n=int(len(hrow)), ckpt=a.ckpt, agreement={}, autofl_truth={})
for k in range(K): res['agreement'][names[k]] = dict(gen_vs_reg=pear(P['gen'][:, k], P['reg'][:, k]), gen_vs_full=pear(P['gen'][:, k], P['full'][:, k]), reg_vs_full=pear(P['reg'][:, k], P['full'][:, k]))
ja = names.index('autofl')
for t in a.targets:
    T = np.load(W + '/' + t, allow_pickle=True); y, m, sl = T['Y'][gidx, ja], T['mask'][gidx, ja], T['slide'][gidx]
    for s in np.unique(sl[m]):
        mm = m & (sl == s); key = f'{t}|slide{int(s)}'
        res['autofl_truth'][key] = {k: pear(P[k][mm, ja], y[mm]) for k in P} | {'n': int(mm.sum())}
np.savez(a.out + '/student_preds.npz', gidx=gidx, hist_row=hrow, **{f'P_{k}': v for k, v in P.items()})
json.dump(res, open(a.out + '/downstream.json', 'w'), indent=1); print(json.dumps(res['autofl_truth'], indent=1))
print('mean agreement gen vs reg (12 markers):', np.mean([res['agreement'][n]['gen_vs_reg'] for n in names[:12]]), ' reg vs full:', np.mean([res['agreement'][n]['reg_vs_full'] for n in names[:12]]))
