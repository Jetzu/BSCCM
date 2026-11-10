"""Cell thumbnails for the deliverable figures: label-free (brightfield, high-NA darkfield, DPC phase) + Giemsa crops of three
confidently classified Giemsa cells (LF cls3 classifier, N008), three MLL23 clinical cells, one virtual stain (N028 cross-slide)."""
import sys, glob, zipfile, io, numpy as np
sys.path.insert(0, '/home/hpc/v181bb/v181bb10/BSCCM')
from bsccm.data import chan_list
from PIL import Image
W = '/hnvme/workspace/v181bb10-bsccm'; out_path = sys.argv[1]
H = np.load(W + '/pack/histology.npy', mmap_mode='r'); hc = np.load(W + '/pack/hist_cells.npy'); L = np.load(W + '/pack/led_array.npy', mmap_mode='r'); D = np.load(W + '/pack/dpc.npy', mmap_mode='r')
P = np.load(glob.glob('/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N008-*/gs_lf_cls3_pred.npz')[0])['P']
def disp(x, lo=1, hi=99.5): a, b = np.percentile(x, [lo, hi]); return np.clip((x - a) / (b - a + 1e-9), 0, 1).astype(np.float32)
out = {}
for name, c in [('gran', 1), ('lym', 0), ('mono', 2)]:
    r = int(np.random.default_rng(3 + c).choice(np.where(P[:, c] > 0.97)[0])); g = int(hc[r]); raw = np.array(L[g]).view(np.int16).astype(np.float32)
    out[f'{name}_bf'] = disp(raw[chan_list('bf')[0]]); out[f'{name}_df'] = disp(raw[chan_list('df_high')].mean(0)); out[f'{name}_dpc'] = disp(np.array(D[g]))
    gs = np.array(H[r]).astype(np.float32); s = 278; o = (398 - s) // 2; gs = gs[o:o + s, o:o + s]; out[f'{name}_gs'] = np.clip(gs / np.percentile(gs, 99.8), 0, 1)
for cls in ['eosinophil', 'neutrophil_segmented', 'lymphocyte_large_granular']:
    z = zipfile.ZipFile(glob.glob(W + f'/external/mll23/{cls}*.zip')[0]); n = [x for x in z.namelist() if x.lower().endswith(('.jpg', '.png', '.tif', '.tiff')) and '__MACOSX' not in x and not x.split('/')[-1].startswith('.')]
    out[f'mll_{cls}'] = np.asarray(Image.open(io.BytesIO(z.read(n[3]))).convert('RGB')).astype(np.float32) / 255.
G = np.load(W + '/runs/n028_vs_batch0/eval/generated.npz'); out['vs_gen'] = G['images'][int(np.random.default_rng(4).choice(len(G['hist_row'])))].astype(np.float32) / 255.
np.savez_compressed(out_path, **out); print('wrote', out_path, len(out))
