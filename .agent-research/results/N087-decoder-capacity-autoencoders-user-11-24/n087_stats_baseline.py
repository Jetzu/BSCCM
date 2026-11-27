"""N087 control for the AE clustering result: hand-made per-channel image statistics (no learning) on the SAME rows and normalisation
as n087_extract.py (GPUNorm + normalise from n087_image_ae). Features per channel (24): mean, std, 10/50/90th percentile over the crop,
mean of the central 32x32 minus mean of the 16-px border (cell-vs-background contrast) = 144 features. Writes a cluster.npz-style file.
usage: python3 -I n087_stats_baseline.py ROWS_NPZ OUT_NPZ [--device cpu|cuda]"""
import sys, argparse, numpy as np, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / 'codex'))
from torch.utils.data import DataLoader
from n087_image_ae import GPUNorm, RawSet, normalise
ap = argparse.ArgumentParser(); ap.add_argument('rows'); ap.add_argument('out'); ap.add_argument('--device', default='cpu'); a = ap.parse_args()
ref = np.load(a.rows); rows, flags, names = ref['G'], ref['is_eval'], ref['names']
norm = GPUNorm(a.device); feats = []; seen = []
for sample in DataLoader(RawSet(rows), batch_size=256, num_workers=8):
    x = normalise(norm, sample, a.device); x = torch.where(torch.isfinite(x), x, 0.).float()
    f = x.flatten(2); q = torch.quantile(f, torch.tensor([.1, .5, .9], device=x.device), dim=2).permute(1, 2, 0)
    c = x[..., 48:80, 48:80].mean((2, 3)); b = torch.cat([x[..., :16, :].flatten(2), x[..., -16:, :].flatten(2)], 2).mean(2)
    feats.append(torch.cat([f.mean(2), f.std(2), q.flatten(1), c - b], 1).cpu().numpy()); seen.append(sample[2].numpy())
H = np.concatenate(feats).astype(np.float32); assert np.array_equal(np.concatenate(seen), rows)
H = (H - H.mean(0)) / (H.std(0) + 1e-6)
np.savez(a.out, H=H.astype(np.float16), G=rows, is_eval=flags, names=names); print('wrote', a.out, H.shape)
