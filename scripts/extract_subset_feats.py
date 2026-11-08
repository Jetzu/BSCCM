"""Frozen SSL-encoder features for LF CHANNEL SUBSETS (absent channels zeroed + presence planes, as in SSL training) -> sim_mosaic inputs.
Subsets: A_bf_phase = brightfield + 4 DPC + phase (morphology, 'pseudo-GS'); B_df<k> = first k darkfield channels of a fixed random
order (k = 1, 2, 4, 8, 17; 'pseudo-pivot' of increasing strength). Rows: n stained cells (any of the 12 markers), all splits.
usage: extract_subset_feats.py --ckpt run/encoder_final.pt --arch resnet50 --out DIR [--n 120000]"""
import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, numpy as np, torch, timm
from bsccm.data import W
from bsccm.gpu_norm import RawSet, GPUNorm
ap = argparse.ArgumentParser(); ap.add_argument('--ckpt', required=True); ap.add_argument('--arch', default='resnet50'); ap.add_argument('--out', required=True)
ap.add_argument('--n', type=int, default=120000); ap.add_argument('--bs', type=int, default=512); ap.add_argument('--workers', type=int, default=15)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
T = np.load(W + '/targets_v3.npz'); rows = np.sort(np.random.default_rng(0).permutation(np.where(T['mask'][:, :12].any(1))[0])[:a.n]); np.save(a.out + '/rows.npy', rows)
df = list(np.random.default_rng(1).permutation(np.arange(1, 18)))
subsets = {'A_bf_phase': [0, 18, 19, 20, 21, 23]}
subsets.update({f'B_df{k:02d}': sorted(int(c) for c in df[:k]) for k in (1, 2, 4, 8, 17)})
model = timm.create_model(a.arch, pretrained=False, in_chans=48, num_classes=0)
model.load_state_dict(torch.load(a.ckpt, map_location='cpu', weights_only=True), strict=True); model = model.cuda().eval().to(memory_format=torch.channels_last)
gn = GPUNorm('cuda'); dl = torch.utils.data.DataLoader(RawSet(rows), batch_size=a.bs, num_workers=a.workers, pin_memory=True, prefetch_factor=6)
masks = {k: torch.zeros(24, device='cuda').index_fill_(0, torch.tensor(v, device='cuda'), 1.0) for k, v in subsets.items()}
out = {k: np.zeros((len(rows), model.num_features), np.float16) for k in subsets}; o = 0
with torch.no_grad():
    for raw, ph, g in dl:
        x = gn(raw.cuda(non_blocking=True), ph.cuda(non_blocking=True), g.cuda(non_blocking=True), 'all', True); B = len(x)
        for k, m in masks.items():
            mm = m[None, :, None, None]; xi = torch.cat([x * mm, mm.expand_as(x)], 1)
            with torch.autocast('cuda', torch.bfloat16): out[k][o:o + B] = model(xi.contiguous(memory_format=torch.channels_last)).float().cpu().numpy()
        o += B
for k, v in out.items(): np.save(f'{a.out}/feats_{k}.npy', v)
import json; json.dump({k: v for k, v in subsets.items()}, open(a.out + '/subsets.json', 'w')); print('done', {k: v.shape for k, v in out.items()})
