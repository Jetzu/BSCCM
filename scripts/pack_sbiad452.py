"""Pack the external S-BIAD452 imaging-flow-cytometry WBC dataset (Lippeveld et al. 2020; ImageStream-X MK II, 60x, 0.33 um/px; one
9-antibody cocktail on EVERY cell) into BSCCM-compatible training arrays (N077/N080).

Inputs (read-only, downloaded from the BioImage Archive): <src>/zarr/<sample>/<part>.zarr.zip (per-cell flattened uint16 stacks, 12 channels
'BF1, CD15, Siglec8, CD14, CD19, SSC, CD3, CD45, BF2, CD4, CD56, CD8'; attrs object_number + shape) and
<src>/original/<sample>/<part>_IDEAS.fcs (compensated per-cell intensities 'Intensity_MC_<marker>', 'Object Number').
Label-free input = BF1 (ch0, 420-480 nm), BF2 (ch8, 570-595 nm), SSC (ch5, side-scatter darkfield):
  BF: log(img / border median) (cf. BSCCM flat-field log ratio); SSC: asinh((img - border median) / max(1.4826 MAD_border, 1)).
  Resized by --scale (default 1.75 = BSCCM/IFC median brightfield cell diameter, same method on both: 73.2 px vs 42.0 px (N080 bfsize); IFC 0.1 um^2/px) and centre-padded/cropped
  to 128 x 128 with 0 (= background after normalisation).
Labels: asinh(I / c_m), c_m = max(1.4826 MAD of the marker over all cells, 1), then z-scored per sample (donor run). Raw I kept.
Outputs (2 files + json, inode-light): <out>/lf.npy float16 (N, 3, 128, 128); <out>/labels.npz (Y float32 (N, 9), I_raw, names, sample,
part, object_number, area_um2, scale, chan_scale = per-channel std of the cell region for input scaling)."""
import argparse, glob, os, struct, sys, json
import numpy as np

MARKERS = ['CD15', 'Siglec8', 'CD14', 'CD19', 'CD3', 'CD45', 'CD4', 'CD56', 'CD8']
FCS_KEY = {'CD15': 'cd15fitc', 'Siglec8': 'siglec8pe', 'CD14': 'cd14pecf594', 'CD19': 'cd19percpcy55', 'CD3': 'cd3bv421',
           'CD45': 'cd45v500', 'CD4': 'cd4bv605', 'CD56': 'cd56apc', 'CD8': 'cd8apccy7'}   # normalised IDEAS column names (runs differ: 'CD15 FITC' / 'CD15_FITC')
LF_CH = [0, 8, 5]   # BF1, BF2, SSC


def read_fcs(path):
    b = open(path, 'rb').read(); hdr = b[:58].decode('ascii')
    ts, te, ds, de = (int(hdr[i:i + 8]) for i in (10, 18, 26, 34))
    txt = b[ts:te + 1].decode('latin-1'); dl = txt[0]; toks, cur, i = [], '', 1
    while i < len(txt):   # FCS 3.x TEXT: a doubled delimiter inside a value is a literal delimiter
        ch = txt[i]
        if ch == dl and i + 1 < len(txt) and txt[i + 1] == dl: cur += dl; i += 2; continue
        if ch == dl: toks.append(cur); cur = ''
        else: cur += ch
        i += 1
    if cur: toks.append(cur)
    kv = dict(zip(toks[0::2], toks[1::2]))
    if kv.get('$DATATYPE') != 'F': raise SystemExit(f'{path}: unsupported FCS datatype {kv.get("$DATATYPE")}')
    n, tot = int(kv['$PAR']), int(kv['$TOT']); names = [kv[f'$P{i}N'] for i in range(1, n + 1)]
    if ds == 0: ds, de = int(kv['$BEGINDATA']), int(kv['$ENDDATA'])
    end = '<' if kv.get('$BYTEORD', '1,2,3,4').startswith('1') else '>'
    data = np.frombuffer(b[ds:ds + 4 * n * tot], dtype=end + 'f4').reshape(tot, n)
    return names, data


def norm(x): return ''.join(ch for ch in x.lower() if ch.isalnum())


def col(names, key):
    nn = [norm(x) for x in names]
    if nn.count(key) != 1: raise SystemExit(f'IDEAS column {key!r}: {nn.count(key)} matches')
    return nn.index(key)


def lf_image(stack, scale):
    import torch, torch.nn.functional as F
    out = []
    for k, c in enumerate(LF_CH):
        x = stack[c].astype(np.float64); border = np.concatenate([x[0], x[-1], x[:, 0], x[:, -1]]); bg = np.median(border)
        if k < 2: y = np.log(np.clip(x / max(bg, 1.0), 1e-3, None))
        else: y = np.arcsinh((x - bg) / max(1.4826 * np.median(np.abs(border - bg)), 1.0))
        out.append(y)
    t = torch.from_numpy(np.stack(out)[None].astype(np.float32))
    h, w = t.shape[-2:]; t = F.interpolate(t, size=(max(1, round(h * scale)), max(1, round(w * scale))), mode='bilinear', align_corners=False)[0]
    res = torch.zeros(3, 128, 128); h, w = t.shape[-2:]
    sy, sx = max(0, (h - 128) // 2), max(0, (w - 128) // 2); t = t[:, sy:sy + 128, sx:sx + 128]; h, w = t.shape[-2:]
    oy, ox = (128 - h) // 2, (128 - w) // 2; res[:, oy:oy + h, ox:ox + w] = t
    return res.numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default='/hnvme/workspace/v181bb10-bsccm/external/sbiad452')
    ap.add_argument('--out', default='/hnvme/workspace/v181bb10-bsccm/external/sbiad452_pack')
    ap.add_argument('--scale', type=float, default=1.75)
    ap.add_argument('--max-cells', type=int, default=0, help='smoke: cells per part')
    ap.add_argument('--allow-partial', action='store_true', help='smoke only: accept fewer than 11 parts')
    a = ap.parse_args()
    import zarr
    parts = sorted(p for p in glob.glob(f'{a.src}/zarr/*/*.zarr.zip') if not p.endswith('.part'))
    if len(parts) != 11 and not a.allow_partial: raise SystemExit(f'expected 11 zarr.zip parts, found {len(parts)}')
    meta, Is, areas = [], [], []
    for pi, zp in enumerate(parts):
        sample = os.path.basename(os.path.dirname(zp)); stem = os.path.basename(zp)[:-len('.zarr.zip')]
        fcs = f'{a.src}/original/{sample}/{stem}_IDEAS.fcs'; names, data = read_fcs(fcs)
        row = {int(o): i for i, o in enumerate(data[:, col(names, 'objectnumber')])}
        g = zarr.open(zarr.storage.ZipStore(zp, mode='r'), mode='r'); obj = g.attrs['object_number']; shp = g.attrs['shape']
        n = len(obj) if not a.max_cells else min(a.max_cells, len(obj))
        keep = [i for i in range(n) if int(obj[i]) in row]
        meta.append((pi, sample, stem, zp, keep, [int(obj[i]) for i in keep], [tuple(shp[i]) for i in keep]))
        rr = [row[int(obj[i])] for i in keep]
        Is.append(np.stack([data[rr, col(names, 'intensitymc' + FCS_KEY[m])] for m in MARKERS], 1)); areas.append(data[rr, col(names, 'aream01')])
        print(f'{stem}: {len(keep)}/{n} cells matched to IDEAS rows', flush=True)
    N = sum(len(m[4]) for m in meta); os.makedirs(a.out, exist_ok=True)
    lf = np.lib.format.open_memmap(f'{a.out}/lf.npy.part', mode='w+', dtype=np.float16, shape=(N, 3, 128, 128)); k = 0
    for pi, sample, stem, zp, keep, objs, shps in meta:
        g = zarr.open(zarr.storage.ZipStore(zp, mode='r'), mode='r')
        for i, s in zip(keep, shps):
            lf[k] = lf_image(np.asarray(g[i]).reshape(s), a.scale); k += 1
        lf.flush(); print(f'packed {stem} -> {k}/{N}', flush=True)
    rs = np.random.default_rng(0).choice(N, min(N, 5000), replace=False)   # per-channel robust scale of the cell region (harness divides by it)
    cen = np.asarray(lf[np.sort(rs), :, 32:96, 32:96], dtype=np.float32); chan_scale = np.array([max(float(np.std(cen[:, c][cen[:, c] != 0])), 1e-3) for c in range(3)], np.float32)
    del lf; os.replace(f'{a.out}/lf.npy.part', f'{a.out}/lf.npy')
    I = np.concatenate(Is).astype(np.float32); c = np.maximum(1.4826 * np.median(np.abs(I - np.median(I, 0)), 0), 1.0)
    Yt = np.arcsinh(I / c); sample = np.concatenate([[m[1]] * len(m[4]) for m in meta]); Y = np.zeros_like(Yt)
    for sv in np.unique(sample):
        s = sample == sv; Y[s] = (Yt[s] - Yt[s].mean(0)) / (Yt[s].std(0) + 1e-6)
    np.savez(f'{a.out}/labels.npz', Y=Y, I_raw=I, cofactor=c, names=np.array(MARKERS), sample=sample,
             part=np.concatenate([[m[2]] * len(m[4]) for m in meta]), object_number=np.concatenate([m[5] for m in meta]),
             area_um2=np.concatenate(areas).astype(np.float32), scale=a.scale, chan_scale=chan_scale)
    json.dump(dict(N=int(N), scale=a.scale, chan_scale=chan_scale.tolist(), cofactor=dict(zip(MARKERS, c.tolist())), parts=[m[2] for m in meta],
                   lf_channels=['BF1 log-ratio', 'BF2 log-ratio', 'SSC asinh'], max_cells=a.max_cells), open(f'{a.out}/pack.json', 'w'), indent=1)
    print('done', N, flush=True)


if __name__ == '__main__':
    main()
