"""BSCCM training-format access. Pack dir holds fixed-record .npy memmaps (row = global_index) + targets.npz.
  led_array (N,23,128,128) u2 | fluor (N,6,128,128) u2 | dpc (N,128,128) f4 | histology (Nh,398,398,3) u2 (rows via targets['hist_row'])"""
import numpy as np, torch, json, os
W = '/hnvme/workspace/v181bb10-bsccm'
LED = ["Brightfield","DF_50","DF_50_Bottom","DF_50_Right","DF_55","DF_60","DF_60_Bottom","DF_60_Right","DF_65","DF_70","DF_70_Bottom","DF_70_Right","DF_75","DF_80","DF_80_Bottom","DF_80_Right","DF_85","DF_90","DPC_Bottom","DPC_Left","DPC_Right","DPC_Top","LED119"]
CHSETS = {'all': list(range(23)), 'bf': [0], 'dpc4': [18, 19, 20, 21], 'bf_dpc4': [0, 18, 19, 20, 21], 'df': list(range(1, 18)),
          'bf_dpc_df60': [0, 5, 18, 19, 20, 21]}
def chan_list(spec):
    if spec in CHSETS: return CHSETS[spec]
    return [int(c) for c in spec.split(',')]            # e.g. "0,18,19"
class LEDMarkerSet(torch.utils.data.Dataset):
    """Label-free image (selected LED channels, log1p + per-channel z-score) -> marker targets/masks + gate class."""
    def __init__(s, rows, channels='all', pack=W + '/pack', stats=W + '/led_stats.npz', train=False):
        s.rows = np.asarray(rows); s.ch = chan_list(channels); s.train = train; s.pack = pack
        s.T = np.load(W + '/targets.npz'); s.m = None
        st = np.load(stats); s.mu = torch.tensor(st['mu'][s.ch]).view(-1, 1, 1); s.sd = torch.tensor(st['sd'][s.ch]).view(-1, 1, 1)
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        if s.m is None: s.m = np.load(s.pack + '/led_array.npy', mmap_mode='r')
        g = s.rows[i]
        x = torch.from_numpy(np.ascontiguousarray(s.m[g][s.ch]).astype(np.float32))
        x = (torch.log1p(x) - s.mu) / s.sd
        if s.train:                                   # D4 augmentation (cells are rotation-invariant)
            k = np.random.randint(4); x = torch.rot90(x, k, (1, 2))
            if np.random.rand() < .5: x = x.flip(2)
        return x, torch.from_numpy(s.T['Y'][g]), torch.from_numpy(s.T['mask'][g]), int(s.T['cls'][g]), int(g)
