#!/usr/bin/env python3 -I
"""Download BSCCM chunks (Dryad v13, 2025-05-06) with resume + sha256 check.
usage: fetch.py <prefix> <outdir> [jobs]   prefix e.g. BSCCM.tar.gz_chunk, BSCCM-tiny.tar.gz_chunk, README.md"""
import json, sys, os, hashlib, subprocess
from concurrent.futures import ThreadPoolExecutor
prefix, out = sys.argv[1], sys.argv[2]; jobs = int(sys.argv[3]) if len(sys.argv) > 3 else 4
man = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dryad_v13_files.json')))['_embedded']['stash:files']
sel = [f for f in man if f['path'].startswith(prefix)]
assert sel, prefix
os.makedirs(out, exist_ok=True)
def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(1 << 24): h.update(b)
    return h.hexdigest()
def token():
    # Dryad API account -> client id + secret -> short-lived bearer token (fetched fresh per attempt)
    if os.environ.get('DRYAD_TOKEN'): return os.environ['DRYAD_TOKEN']
    r = subprocess.run(['curl', '-sSf', '-X', 'POST', 'https://datadryad.org/oauth/token',
        '-d', 'client_id=' + os.environ['DRYAD_CLIENT_ID'], '-d', 'client_secret=' + os.environ['DRYAD_CLIENT_SECRET'],
        '-d', 'grant_type=client_credentials', '-d', 'redirect_uri=oob'], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)['access_token']
def get(f):
    p = os.path.join(out, f['path']); url = 'https://datadryad.org' + f['_links']['stash:download']['href']
    for attempt in range(8):
        if os.path.exists(p) and os.path.getsize(p) == f['size'] and sha(p) == f['digest']:
            print('OK', f['path'], flush=True); return True
        if os.path.exists(p) and (os.path.getsize(p) > f['size'] or os.path.getsize(p) < 1024): os.remove(p)  # error bodies / overshoot
        hdr = ['-H', 'Authorization: Bearer ' + token()]
        subprocess.run(['curl', '-sSfL', '-C', '-', '--retry', '5', '--retry-delay', '10', *hdr, '-o', p, url])
    print('FAIL', f['path'], flush=True); return False
with ThreadPoolExecutor(jobs) as ex: res = list(ex.map(get, sel))
sys.exit(0 if all(res) else 1)
