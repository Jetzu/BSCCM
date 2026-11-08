import sys, os, time; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from bsccm.data import LFNorm; from bsccm.gpu_norm import GPUNorm, RawSet
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
meta = np.load('/hnvme/workspace/v181bb10-bsccm/cellmeta.npz'); rows = np.r_[np.random.default_rng(0).choice(412941, 30, replace=False), np.argmin(meta['pos_x']), np.argmax(meta['pos_y'])]
cpu = LFNorm(); gpu = GPUNorm(dev); ds = RawSet(rows)
b = [ds[i] for i in range(len(rows))]; raw = torch.stack([x[0] for x in b]).to(dev); ph = torch.stack([x[1] for x in b]).to(dev); g = torch.tensor([x[2] for x in b], device=dev)
xg = gpu(raw, ph, g, 'all', True).cpu().numpy(); xc = np.stack([cpu(int(r), list(range(23)), True) for r in rows])
print('max abs diff', float(np.abs(xg - xc).max()), 'rel', float(np.abs(xg - xc).max() / np.abs(xc).max()))
t = time.time(); [ds[i] for i in range(len(rows))] ; print('worker raw-copy ms/cell (warm):', (time.time() - t) / len(rows) * 1e3)
