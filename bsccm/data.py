"""BSCCM training-format access (pack = fixed-record .npy memmaps, row = global_index).
  led_array (N,23,128,128) u2 | fluor (N,6,128,128) u2 [NEVER an LF input: contains the marker signal] | dpc (N,128,128) f4 (quantitative phase, rad)
  histology (Nh,398,398,3) u2 (rows via targets_v2['hist_row'])
LF normalisation v2 (N005): ratio = (raw-30)/(bg_p50 crop at the cell's FOV position - 30); x = log(ratio / slide_scale[slide,c]) / s_c.
Phase channel appended (optional): dpc / 1.0 rad."""
import numpy as np, torch
W = '/hnvme/workspace/v181bb10-bsccm'
LED = ["Brightfield","DF_50","DF_50_Bottom","DF_50_Right","DF_55","DF_60","DF_60_Bottom","DF_60_Right","DF_65","DF_70","DF_70_Bottom","DF_70_Right","DF_75","DF_80","DF_80_Bottom","DF_80_Right","DF_85","DF_90","DPC_Bottom","DPC_Left","DPC_Right","DPC_Top","LED119"]
CHSETS = {'all': list(range(23)), 'bf': [0], 'dpc4': [18, 19, 20, 21], 'bf_dpc4': [0, 18, 19, 20, 21], 'df': list(range(1, 18)), 'no119': list(range(22))}
def chan_list(spec):
    if spec in CHSETS: return CHSETS[spec]
    return [int(c) for c in spec.split(',')]
class LFNorm:
    def __init__(s, pack=W + '/pack'):
        s.pack = pack; s.led = None
        meta = np.load(W + '/cellmeta.npz'); s.py, s.px, s.slide = meta['pos_y'], meta['pos_x'], meta['slide']
        n = np.load(W + '/led_norm.npz'); s.scale, s.s_c = n['slide_scale'], n['s_c']
    def _open(s):
        if s.led is None:
            s.led = np.load(s.pack + '/led_array.npy', mmap_mode='r'); s.bg = np.load(W + '/backgrounds_p50.npy', mmap_mode='r'); s.dpc = np.load(s.pack + '/dpc.npy', mmap_mode='r')
    def __call__(s, g, ch, phase=False):
        s._open(); H = s.bg.shape[1]
        ys = np.clip(np.arange(s.py[g] - 64, s.py[g] + 64), 0, H - 1); xs = np.clip(np.arange(s.px[g] - 64, s.px[g] + 64), 0, H - 1)
        raw = np.asarray(s.led[g][ch], np.float32); b = np.asarray(s.bg[ch][:, ys][:, :, xs], np.float32)
        r = (raw - 30) / (b - 30) / s.scale[s.slide[g], ch][:, None, None]
        x = np.log(np.clip(r, 1e-3, None)) / s.s_c[ch][:, None, None]
        if phase: x = np.concatenate([x, np.asarray(s.dpc[g], np.float32)[None]], 0)
        return x
class CellSet(torch.utils.data.Dataset):
    """LF image (+ optional phase) -> marker targets/masks (targets_v2), class labels, global row."""
    def __init__(s, rows, channels='all', phase=True, train=False, targets=W + '/targets_v2.npz'):
        s.rows = np.asarray(rows); s.ch = chan_list(channels); s.phase = phase; s.train = train; s.norm = LFNorm()
        T = np.load(targets); s.Y, s.M, s.c3, s.c10 = T['Y'], T['mask'], T['cls3'], T['cls10']
    def __len__(s): return len(s.rows)
    def __getitem__(s, i):
        g = int(s.rows[i]); x = torch.from_numpy(s.norm(g, s.ch, s.phase))
        if s.train:   # NO rotations/flips: DPC_Left/Right/Top/Bottom and DF_*_Bottom/_Right are direction-specific, so a rotation
            # would silently relabel illumination directions. Only small translations (cells are centred +- detection error).
            dy, dx = np.random.randint(-4, 5, 2); x = torch.roll(x, (int(dy), int(dx)), (1, 2))
        return x, torch.from_numpy(s.Y[g]), torch.from_numpy(s.M[g]), int(s.c3[g]), int(s.c10[g]), g
