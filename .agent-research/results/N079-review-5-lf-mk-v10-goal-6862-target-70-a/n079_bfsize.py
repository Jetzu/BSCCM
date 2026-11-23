"""Same-method cell diameter (native px) on brightfield: |log(img/border median)| smoothed (sigma 1.5), > border median + 5 x 1.4826 MAD,
component at/near centre, holes filled -> equivalent diameter. BSCCM: Brightfield LED (index 0) of 600 stained cells; IFC: BF1 (ch0) of 600 cells."""
import numpy as np, zarr, glob
from scipy import ndimage as ndi
def diam(img):
    x = img.astype(np.float64); b = np.concatenate([x[0], x[-1], x[:, 0], x[:, -1]]); y = np.abs(np.log(np.clip(x / np.median(b), 1e-3, None)))
    y = ndi.gaussian_filter(y, 1.5); yb = np.concatenate([y[0], y[-1], y[:, 0], y[:, -1]]); thr = np.median(yb) + 5 * 1.4826 * np.median(np.abs(yb - np.median(yb)))
    lab, n = ndi.label(y > thr)
    if n == 0: return np.nan
    cy, cx = np.array(y.shape) // 2; ids = lab[cy - 3:cy + 4, cx - 3:cx + 4].ravel(); ids = ids[ids > 0]
    c = np.bincount(ids).argmax() if len(ids) else np.bincount(lab.ravel())[1:].argmax() + 1
    return 2 * np.sqrt(ndi.binary_fill_holes(lab == c).sum() / np.pi)
W = '/hnvme/workspace/v181bb10-bsccm'; T = np.load(W + '/targets_v5cx_v10.npz'); led = np.load(W + '/pack/led_array.npy', mmap_mode='r')
meta = np.load(W + '/cellmeta.npz'); bg = np.load(W + '/backgrounds_p50.npy', mmap_mode='r')
rng = np.random.default_rng(0); G = np.sort(rng.choice(np.flatnonzero(T['stained']), 600, replace=False)); db = []
for g in G:
    raw = np.asarray(led[g][0], np.float64); y0, x0 = int(meta['pos_y'][g]) - 64, int(meta['pos_x'][g]) - 64
    if y0 < 0 or x0 < 0 or y0 + 128 > bg.shape[1] or x0 + 128 > bg.shape[2]: continue
    ff = np.asarray(bg[0, y0:y0 + 128, x0:x0 + 128], np.float64); db.append(diam((raw - 30) / (ff - 30) * 1000))
zp = sorted(p for p in glob.glob(W + '/external/sbiad452/zarr/*/*.zarr.zip'))[0]; g = zarr.open(zarr.storage.ZipStore(zp, mode='r'), mode='r'); sh = g.attrs['shape']
di = [diam(np.asarray(g[i]).reshape(sh[i])[0]) for i in range(0, 6000, 10)]
db, di = np.array(db), np.array(di)
print('BSCCM BF diam px median %.1f IQR %.1f-%.1f n %d' % (np.nanmedian(db), *np.nanpercentile(db, [25, 75]), np.isfinite(db).sum()))
print('IFC   BF1 diam px median %.1f IQR %.1f-%.1f n %d' % (np.nanmedian(di), *np.nanpercentile(di, [25, 75]), np.isfinite(di).sum()))
print('scale BSCCM/IFC = %.3f' % (np.nanmedian(db) / np.nanmedian(di)))
