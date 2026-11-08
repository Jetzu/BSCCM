#!/usr/bin/env python3
"""Stage 1: scan the zip central directory once -> compact numpy offset tables of every zarr chunk (data offset, size).
usage: build_index.py <BSCCM.zarr.zip> <out_index.npz>"""
import sys, zipfile, struct, json, hashlib, collections, random
import numpy as np
zp, out = sys.argv[1:3]
N = 412941
z = zipfile.ZipFile(zp)
infos = z.infolist()
print(len(infos), 'entries', flush=True)
shape = {'led_array': (N, 23), 'fluor': (N, 6), 'dpc': (N, 1)}
off = {k: np.zeros(v, np.uint64) for k, v in shape.items()}; sz = {k: np.zeros(v, np.uint32) for k, v in shape.items()}
hist_cells = []; hoff = {}; hsz = {}
zarr_meta = collections.defaultdict(collections.Counter)  # modality -> Counter(.zarray bytes)
nameadd = {}
extras = collections.Counter()
rng = random.Random(0)
chunk_rows = []   # (kind, cell, ch, header_offset, size, namelen)
for zi in infos:
    p = zi.filename.split('/')
    if len(p) < 7 or p[1] != 'BSCCM_images.zarr': continue
    mod, cell, key = p[-3], p[-2], p[-1]
    if key.startswith('.'):
        if key == '.zarray': zarr_meta[mod][z.read(zi)] += 1
        continue
    g = int(cell[5:]); ks = key.split('.')
    if mod == 'histology':
        chunk_rows.append((g, int(ks[0]) * 2 + int(ks[1]), zi.header_offset, zi.file_size, len(zi.filename.encode())))
    else:
        c = int(ks[0]); assert mod in shape
        off[mod][g, c] = zi.header_offset; sz[mod][g, c] = zi.file_size
        nameadd[(mod, g, c)] = None if False else len(zi.filename.encode())
print('chunk scan done', flush=True)
# uniform metadata check
meta = {}
for mod, cnt in zarr_meta.items():
    print(mod, 'distinct .zarray variants:', len(cnt), flush=True)
    meta[mod] = [json.loads(b) for b, _ in cnt.most_common()]
# histology tables
hc = sorted({r[0] for r in chunk_rows}); hrow = {g: i for i, g in enumerate(hc)}
h_off = np.zeros((len(hc), 4), np.uint64); h_sz = np.zeros((len(hc), 4), np.uint32)
for g, k, ho, s, nl in chunk_rows:
    h_off[hrow[g], k] = ho; h_sz[hrow[g], k] = s
# convert header offsets -> data offsets; namelen needed: derive from the name format (cell number digits vary) -> recompute
# simpler: store header offsets; worker adds 30+name+extra by reading local header (2 preads). Check extra-length constancy:
import os
with open(zp, 'rb') as f:
    samp = rng.sample([i for i in infos if '.zarr/' in i.filename and not i.filename.split('/')[-1].startswith('.')], 5000)
    for zi in samp:
        f.seek(zi.header_offset); h = f.read(30); assert h[:4] == b'PK\x03\x04'
        fn, ex = struct.unpack('<HH', h[26:30]); assert fn == len(zi.filename.encode())
        extras[ex] += 1
print('local extra-length distribution', dict(extras), flush=True)
extra = extras.most_common(1)[0][0]; assert len(extras) == 1
np.savez(out, led_off=off['led_array'], led_sz=sz['led_array'], fl_off=off['fluor'], fl_sz=sz['fluor'],
         dpc_off=off['dpc'], dpc_sz=sz['dpc'], h_cells=np.array(hc), h_off=h_off, h_sz=h_sz, local_extra=np.array(extra))
json.dump(meta, open(out.replace('.npz', '_zarrmeta.json'), 'w'), indent=1)
for k in ('led_array', 'fluor', 'dpc'):
    print(k, 'missing chunks:', int((sz[k] == 0).sum()), 'of', sz[k].size)
print('histology cells', len(hc), 'missing chunks', int((h_sz == 0).sum()))
