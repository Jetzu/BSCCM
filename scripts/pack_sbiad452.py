"""Pack the external S-BIAD452 imaging-flow-cytometry WBC dataset (Lippeveld et al. 2020; ImageStream-X MK II, 60x, 0.33 um/px; one
9-antibody cocktail on EVERY cell) into BSCCM-compatible training arrays (N077/N080).

Inputs (read-only, downloaded from the BioImage Archive): <src>/zarr/<sample>/<part>.zarr.zip (per-cell flattened uint16 stacks, 12 channels
'BF1, CD15, Siglec8, CD14, CD19, SSC, CD3, CD45, BF2, CD4, CD56, CD8'; attrs object_number + shape) and
<src>/original/<sample>/<part>_IDEAS.fcs (compensated per-cell intensities 'Intensity_MC_<marker>', 'Object Number').
Label-free input = BF1 (ch0, 420-480 nm), BF2 (ch8, 570-595 nm), SSC (ch5, side-scatter darkfield):
  BF: log(img / border median) (cf. BSCCM flat-field log ratio); SSC: asinh((img - border median) / max(1.4826 MAD_border, 1)).
  Resized by --scale (default 1.75 = BSCCM/IFC median brightfield cell diameter, same method on both: 73.2 px vs 42.0 px (N080 bfsize); IFC 0.1 um^2/px) and centre-padded/cropped
  to 128 x 128 with 0 (= background after normalisation).
Labels: asinh(I / c_m), c_m = max(1.4826 MAD of the marker over --stat-donors cells, 1), then z-scored per sample (donor run). Raw I kept.
Outputs (2 files + json, inode-light): <out>/lf.npy float16 (N, 3, 128, 128); <out>/labels.npz (Y float32 (N, 9), I_raw, names, sample,
part, object_number, area_um2, scale, chan_scale = per-channel std of the cell region for input scaling)."""
import argparse, glob, os, struct, sys, json, re
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


def donor_name(run):
    match = re.match(r'^sample\d+', str(run))
    if match is None: raise ValueError(f'invalid IFC run name: {run!r}')
    return match.group(0)


def statistic_rows(sample, stat_donors):
    donors = np.array([donor_name(run) for run in sample])
    requested = sorted({d.strip() for d in stat_donors.split(',') if d.strip()})
    if not requested or any(re.fullmatch(r'sample\d+', d) is None for d in requested):
        raise ValueError('--stat-donors must contain DONOR names, e.g. sample0,sample1')
    if set(requested) - set(donors): raise ValueError('--stat-donors contains donors missing from the pack')
    rows = np.flatnonzero(np.isin(donors, requested))
    if not len(rows): raise ValueError('--stat-donors has no cells')
    return requested, rows


def fit_statistics(lf, intensity, stat_rows):
    # Only eligible donors enter either fit, including the deterministic input subsample.
    rs = np.random.default_rng(0).choice(stat_rows, min(len(stat_rows), 5000), replace=False)
    cen = np.asarray(lf[np.sort(rs), :, 32:96, 32:96], dtype=np.float32)
    scales = []
    for channel in range(3):
        values = cen[:, channel][cen[:, channel] != 0]
        sd = float(np.std(values)) if len(values) else 0.0
        if not np.isfinite(sd): raise ValueError('nonfinite channel statistic')
        scales.append(max(sd, 1e-3))
    train_i = intensity[stat_rows]
    cofactor = np.maximum(1.4826 * np.median(np.abs(train_i - np.median(train_i, 0)), 0), 1.0)
    if not np.isfinite(cofactor).all(): raise ValueError('nonfinite marker cofactor')
    return np.array(scales, np.float32), cofactor, np.sort(rs)


# --- N080/N081 (2026-11-24): sample2 parts on the archive lack .zarray/.zattrs (chunk files only) -> rebuild the array metadata from a
# metadata-bearing part (same writer: chunks 100, blosc-lz4, vlen-array <u2) and infer each cell's (H, W) from its flattened 12*H*W length
# by brightfield row smoothness (validated 2084/2084 exact on sample0 cells with known shapes). Labels: image-derived intensities for ALL
# donors (one definition): per fluorescence channel, sum of (pixel - border median) inside the brightfield cell mask (log-ratio |.|,
# gaussian 1.5, > border median + 5 x 1.4826 MAD, holes filled, eroded 2 px); asinh agreement with IDEAS Intensity_MC on sample0:
# CD15 1.00, Siglec8 .977, CD14 .948, CD19 .762, CD3 .998, CD45 .997, CD4 .966, CD56 .937, CD8 .908 (mean .943). IDEAS values are kept
# (I_ideas; NaN where no object number) for a sensitivity check.
FLUOR_CH = {'CD15': 1, 'Siglec8': 2, 'CD14': 3, 'CD19': 4, 'CD3': 6, 'CD45': 7, 'CD4': 9, 'CD56': 10, 'CD8': 11}


def infer_hw(flat, wmin=30, wmax=120, hmin=30, hmax=160):
    n = flat.size // 12; bf = flat[:n].astype(np.float64); best = None
    for w in range(wmin, wmax + 1):
        if n % w: continue
        h = n // w
        if not (hmin <= h <= hmax): continue
        sc = np.abs(np.diff(bf.reshape(h, w), axis=0)).mean()
        if best is None or sc < best[0]: best = (sc, h, w)
    if best is None: raise ValueError(f'no plausible (H, W) for length {flat.size}')
    return (12, best[1], best[2])


def image_intensity(st):
    from scipy import ndimage as ndi
    bf = st[0].astype(np.float64); b = np.concatenate([bf[0], bf[-1], bf[:, 0], bf[:, -1]])
    y = ndi.gaussian_filter(np.abs(np.log(np.clip(bf / max(np.median(b), 1.0), 1e-3, None))), 1.5)
    yb = np.concatenate([y[0], y[-1], y[:, 0], y[:, -1]]); m = y > np.median(yb) + 5 * 1.4826 * np.median(np.abs(yb - np.median(yb)))
    m = ndi.binary_erosion(ndi.binary_fill_holes(m), iterations=2)
    out = []
    for k in MARKERS:
        x = st[FLUOR_CH[k]].astype(np.float64); bb = np.concatenate([x[0], x[-1], x[:, 0], x[:, -1]]); x = x - np.median(bb)
        out.append(x[m].sum() if m.any() else x.sum())
    return out


def open_part(zp, ref_meta):
    import zarr, zipfile, json as _j
    zf = zipfile.ZipFile(zp); names = zf.namelist()
    if '.zarray' in names:
        g = zarr.open(zarr.storage.ZipStore(zp, mode='r'), mode='r')   # materialise once: per-cell indexing would decode a 100-cell chunk per access
        return g[:], list(g.attrs['object_number']), [tuple(x) for x in g.attrs['shape']]
    keys = sorted(int(k) for k in names); meta = dict(ref_meta); meta['shape'] = [(max(keys) + 1) * meta['chunks'][0]]
    store = {'.zarray': _j.dumps(meta).encode()}; store.update({str(k): zf.read(str(k)) for k in keys})
    arr = zarr.open(store, mode='r')[:]; n = 0   # object fill value is 0 -> stop at the first non-array / too-short element
    while n < len(arr) and isinstance(arr[n], np.ndarray) and arr[n].size >= 12 and arr[n].size % 12 == 0: n += 1
    return arr[:n], [-1] * n, None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--src', default='/hnvme/workspace/v181bb10-bsccm/external/sbiad452')
    ap.add_argument('--out', default='/hnvme/workspace/v181bb10-bsccm/external/sbiad452_pack')
    ap.add_argument('--scale', type=float, default=1.75)
    ap.add_argument('--stat-donors', default='sample0,sample1', help='DONOR names used to fit cofactors and chan_scale')
    ap.add_argument('--max-cells', type=int, default=0, help='smoke: cells per part')
    ap.add_argument('--allow-partial', action='store_true', help='smoke only: accept fewer than 11 parts')
    a = ap.parse_args()
    import zarr
    parts = sorted(p for p in glob.glob(f'{a.src}/zarr/*/*.zarr.zip') if not p.endswith('.part'))
    if len(parts) != 11 and not a.allow_partial: raise SystemExit(f'expected 11 zarr.zip parts, found {len(parts)}')
    meta, Is, Iid, areas = [], [], [], []
    import zipfile
    ref = next(json.loads(zipfile.ZipFile(p).read('.zarray')) for p in parts if '.zarray' in zipfile.ZipFile(p).namelist())
    for pi, zp in enumerate(parts):
        sample = os.path.basename(os.path.dirname(zp)); stem = os.path.basename(zp)[:-len('.zarr.zip')]
        fcs = f'{a.src}/original/{sample}/{stem}_IDEAS.fcs'; names, data = read_fcs(fcs)
        row = {int(o): i for i, o in enumerate(data[:, col(names, 'objectnumber')])}
        g, obj, shp = open_part(zp, ref)
        n = len(obj) if not a.max_cells else min(a.max_cells, len(obj))
        keep = list(range(n)) if shp is None else [i for i in range(n) if int(obj[i]) in row]
        shapes = [infer_hw(np.asarray(g[i])) if shp is None else shp[i] for i in keep]
        meta.append((pi, sample, stem, zp, keep, [int(obj[i]) for i in keep], shapes, g))
        Is.append(np.array([image_intensity(np.asarray(g[i]).reshape(sh)) for i, sh in zip(keep, shapes)], np.float64))
        if shp is None:
            Iid.append(np.full((len(keep), 9), np.nan)); areas.append(np.full(len(keep), np.nan))
        else:
            rr = [row[int(obj[i])] for i in keep]
            Iid.append(np.stack([data[rr, col(names, 'intensitymc' + FCS_KEY[m])] for m in MARKERS], 1)); areas.append(data[rr, col(names, 'aream01')])
        print(f"{stem}: {len(keep)}/{n} cells ({'metadata rebuilt, shapes inferred, no IDEAS link' if shp is None else 'matched to IDEAS rows'})", flush=True)
    N = sum(len(m[4]) for m in meta)
    sample = np.concatenate([[m[1]] * len(m[4]) for m in meta])
    stat_donors, stat_rows = statistic_rows(sample, a.stat_donors)
    os.makedirs(a.out, exist_ok=True)
    lf = np.lib.format.open_memmap(f'{a.out}/lf.npy.part', mode='w+', dtype=np.float16, shape=(N, 3, 128, 128)); k = 0
    for pi, run, stem, zp, keep, objs, shps, g in meta:
        for i, s in zip(keep, shps):
            lf[k] = lf_image(np.asarray(g[i]).reshape(s), a.scale); k += 1
        lf.flush(); print(f'packed {stem} -> {k}/{N}', flush=True)
    I = np.concatenate(Is).astype(np.float32); I_ideas = np.concatenate(Iid).astype(np.float32)
    chan_scale, c, scale_rows = fit_statistics(lf, I, stat_rows)
    del lf; os.replace(f'{a.out}/lf.npy.part', f'{a.out}/lf.npy')
    Yt = np.arcsinh(I / c); Y = np.zeros_like(Yt)
    for sv in np.unique(sample):
        s = sample == sv; Y[s] = (Yt[s] - Yt[s].mean(0)) / (Yt[s].std(0) + 1e-6)
    np.savez(f'{a.out}/labels.npz', Y=Y, I_raw=I, I_ideas=I_ideas, cofactor=c, names=np.array(MARKERS), sample=sample,
             part=np.concatenate([[m[2]] * len(m[4]) for m in meta]), object_number=np.concatenate([m[5] for m in meta]),
             area_um2=np.concatenate(areas).astype(np.float32), scale=a.scale, chan_scale=chan_scale)
    json.dump(dict(N=int(N), scale=a.scale, chan_scale=chan_scale.tolist(), cofactor=dict(zip(MARKERS, c.tolist())), parts=[m[2] for m in meta],
                   stat_donors=stat_donors, n_stat_cells=len(stat_rows), n_chan_scale_cells=len(scale_rows),
                   stat_runs=sorted(set(sample[stat_rows])),
                   label_standardization='per run; cofactors fitted on stat_donors only',
                   label_source='image-derived masked intensities for all donors (IDEAS kept as I_ideas)',
                   lf_channels=['BF1 log-ratio', 'BF2 log-ratio', 'SSC asinh'], max_cells=a.max_cells), open(f'{a.out}/pack.json', 'w'), indent=1)
    print('done', N, flush=True)


if __name__ == '__main__':
    main()
