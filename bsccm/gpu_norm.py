"""Fast LF pipeline: workers only copy raw uint16 records (as int16, values <= 4095) + phase + cell meta; the flat-field / slide-scale / log
normalisation runs batched on the GPU. Numerically identical to bsccm.data.LFNorm (checked in scripts/check_gpu_norm.py).
  raw (B,23,128,128) int16, dpc (B,128,128) f32, g (B,) global rows  ->  x (B,C,128,128) f32"""
import numpy as np, torch
from bsccm.data import W, chan_list
class RawSet(torch.utils.data.Dataset):
    """Returns raw LED record (int16 view, no float maths in the worker), phase image, global row, and arbitrary per-row label arrays."""
    def __init__(s, rows, labels=None, pack=W + '/pack'):
        s.rows = np.asarray(rows); s.pack = pack; s.led = None; s.labels = labels or {}
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        if s.led is None: s.led = np.load(s.pack + '/led_array.npy', mmap_mode='r'); s.dpc = np.load(s.pack + '/dpc.npy', mmap_mode='r')
        g = int(s.rows[i]); raw = np.array(s.led[g]).view(np.int16); ph = np.array(s.dpc[g])
        return (torch.from_numpy(raw), torch.from_numpy(ph), g) + tuple(torch.as_tensor(v[g]) for v in s.labels.values())
class GPUNorm:
    def __init__(s, device='cuda'):
        meta = np.load(W + '/cellmeta.npz'); n = np.load(W + '/led_norm.npz'); d = device
        s.py = torch.tensor(meta['pos_y'], dtype=torch.long, device=d); s.px = torch.tensor(meta['pos_x'], dtype=torch.long, device=d)
        s.slide = torch.tensor(meta['slide'], dtype=torch.long, device=d)
        s.scale = torch.tensor(n['slide_scale'], device=d); s.s_c = torch.tensor(n['s_c'], device=d)
        s.bg = torch.tensor(np.load(W + '/backgrounds_p50.npy'), device=d) - 30.0       # (23,H,W) f32, offset removed
        s.H = s.bg.shape[1]; s.ar = torch.arange(128, device=d)
    @torch.no_grad()
    def __call__(s, raw, dpc, g, ch='all', phase=True, shift=0):
        """raw int16 (B,23,128,128) on device; g LongTensor (B,) on device. shift>0: random translation +-shift px (training aug, applied
        to the normalised image so raw and flat field stay registered)."""
        c = torch.tensor(chan_list(ch) if isinstance(ch, str) else list(ch), dtype=torch.long, device=raw.device)
        ys = (s.py[g][:, None] - 64 + s.ar).clamp(0, s.H - 1); xs = (s.px[g][:, None] - 64 + s.ar).clamp(0, s.H - 1)   # edge-pad == clamp
        bgc = s.bg[c[None, :, None, None], ys[:, None, :, None], xs[:, None, None, :]]                              # (B,C,128,128) gather, no full-frame copy
        r = (raw[:, c].float() - 30.0) / bgc / s.scale[s.slide[g]][:, c][:, :, None, None]
        x = torch.log(r.clamp(min=1e-3)) / s.s_c[c][None, :, None, None]
        if phase: x = torch.cat([x, dpc[:, None].float()], 1)
        if shift:
            dy, dx = torch.randint(-shift, shift + 1, (2, len(x)))
            x = torch.stack([torch.roll(x[i], (int(dy[i]), int(dx[i])), (1, 2)) for i in range(len(x))])
        return x
