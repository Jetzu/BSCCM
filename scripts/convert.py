#!/usr/bin/env python3 -I
"""Stream BSCCM chunks (concatenated tar.gz) into ONE zarr-readable zip (no loose files, no inodes).
usage: convert.py <chunk_dir> <chunk_prefix> <out.zip>
Entries are stored uncompressed (zarr chunks are already Blosc/zstd). Open with zarr.storage.ZipStore(out.zip)
(zarr group 'BSCCM_images.zarr/...' etc.); csv/json ride along as entries. Tar root dir is kept as path prefix."""
import sys, os, glob, tarfile, zipfile, io, shutil
d, prefix, out = sys.argv[1:4]
chunks = sorted(glob.glob(os.path.join(d, prefix + '*')))
assert chunks, 'no chunks'
class Cat(io.RawIOBase):
    def __init__(s, fs): s.fs, s.i, s.f = fs, 0, None
    def readable(s): return True
    def readinto(s, b):
        while True:
            if s.f is None:
                if s.i >= len(s.fs): return 0
                s.f = open(s.fs[s.i], 'rb'); s.i += 1
            n = s.f.readinto(b)
            if n: return n
            s.f.close(); s.f = None
tmp = out + '.part'
n = nbytes = 0
with zipfile.ZipFile(tmp, 'w', zipfile.ZIP_STORED, allowZip64=True) as z, \
     tarfile.open(fileobj=io.BufferedReader(Cat(chunks), 1 << 24), mode='r|gz') as t:
    for m in t:
        if m.isdir(): continue
        if not m.isfile(): print('skip', m.name, m.type); continue
        with z.open(zipfile.ZipInfo(m.name, date_time=(1980, 1, 1, 0, 0, 0)), 'w', force_zip64=True) as w:
            shutil.copyfileobj(t.extractfile(m), w, 1 << 22)
        n += 1; nbytes += m.size
        if n % 20000 == 0: print(n, 'entries', nbytes / 1e9, 'GB', flush=True)
os.rename(tmp, out)
print('done', n, 'entries', nbytes / 1e9, 'GB ->', out)
