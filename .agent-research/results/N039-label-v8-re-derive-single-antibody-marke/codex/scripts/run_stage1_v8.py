"""Reproduce N039 stage 1 without editing the repo or read-only data.

Use the pinned environment in requirements.txt. Copies the three existing
pipeline scripts into /tmp, applies pipeline.diff there, builds raw/calibrated
labels, and writes only QC and small audit artifacts to --out. The rejected
candidate stays in scratch/targets; never train/score on it after QC failure.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(repo, scratch, out, proxy_as_primary=False):
    repo, scratch, out = Path(repo).resolve(), Path(scratch).resolve(), Path(out).resolve()
    bundle = Path(__file__).resolve().parent.parent
    if not str(scratch).startswith('/tmp/codex_'):
        raise ValueError('Scratch must be under /tmp/codex_<item>/')
    if out == repo or out == Path('/hnvme/workspace/v181bb10-bsccm'):
        raise ValueError('Use a separate writable output directory')
    for p in (scratch, out, scratch / 'pipeline', scratch / 'targets', scratch / 'tmp'):
        p.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, TMPDIR=str(scratch / 'tmp'),
               PYTHONPYCACHEPREFIX=str(scratch / 'pycache'),
               BSCCM_RAW='v8', BSCCM_CLEAN='1', BSCCM_DETREND='xfit',
               BSCCM_TARGET_DIR=str(scratch / 'targets'),
               BSCCM_V8_RAW=str(scratch / 'targets' / 'raw_v8.parquet'),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for version in (3, 4, 5):
        name = f'make_targets_v{version}.py'
        shutil.copy2(repo / 'scripts' / name, scratch / 'pipeline' / name)
    with (bundle / 'pipeline.diff').open() as patch:
        subprocess.run(['patch', '--batch', '-p2', '-d', str(scratch / 'pipeline')],
                       stdin=patch, env=env, check=True)
    calls = [('raw', [str(bundle / 'scripts' / 'make_targets_v8_raw.py'), '--out', str(scratch / 'targets')]),
             ('v3', [str(scratch / 'pipeline' / 'make_targets_v3.py')]),
             ('v4', [str(scratch / 'pipeline' / 'make_targets_v4.py'), str(scratch / 'targets')]),
             ('v5', [str(scratch / 'pipeline' / 'make_targets_v5.py')]),
             ('qc', [str(bundle / 'scripts' / 'qc_targets_v8.py'), '--targets',
                     str(scratch / 'targets' / 'targets_v8cx.npz'), '--out', str(out)] +
                    (['--proxy-as-primary'] if proxy_as_primary else []))]
    for name, args in calls:
        print('Running', name, flush=True)
        with (scratch / f'{name}.log').open('w') as log, (scratch / f'{name}.err').open('w') as err:
            subprocess.run([sys.executable] + args, env=env, stdout=log, stderr=err, check=True)
    for name in ('unmixing_maps.json', 'unmixing_fidelity.csv',
                 'targets_v4x_detrend.json', 'targets_v4x_detrend_summary.csv'):
        shutil.copy2(scratch / 'targets' / name, out / name)
    for name in ('raw', 'v3', 'v4', 'v5'):
        shutil.copy2(scratch / f'{name}.log', out / f'{name}_build.log')
    import numpy, pandas, scipy, sklearn, pyarrow
    versions = {m.__name__: m.__version__ for m in (numpy, pandas, scipy, sklearn, pyarrow)}
    hashes = {name: hashlib.sha256((scratch / 'targets' / name).read_bytes()).hexdigest()
              for name in ('raw_v8.parquet', 'targets_v8cx.npz')}
    (out / 'build_manifest.json').write_text(json.dumps(dict(
        versions=versions, repo=str(repo), scratch=str(scratch), hashes=hashes,
        config={k: env[k] for k in ('BSCCM_RAW', 'BSCCM_CLEAN', 'BSCCM_DETREND')},
        qc=json.loads((out / 'qc_v8_meta.json').read_text()), model_scoring=False), indent=2) + '\n')
    status = json.loads((out / 'qc_v8_meta.json').read_text())['overall']
    print(f'QC {status}; candidate retained only in {scratch / "targets"}', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default='/home/hpc/v181bb/v181bb10/BSCCM')
    ap.add_argument('--scratch', default='/tmp/codex_N039')
    ap.add_argument('--out', required=True)
    ap.add_argument('--proxy-as-primary', action='store_true', help='Only with explicit user authorization')
    a = ap.parse_args()
    run(a.repo, a.scratch, a.out, a.proxy_as_primary)
