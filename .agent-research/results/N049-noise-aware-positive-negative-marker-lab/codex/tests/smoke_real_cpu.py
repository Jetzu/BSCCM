"""Two CPU optimiser steps on 8 actual LF cells; evaluate 24 held-out-draw cells.

Uses the delivered draw-0 archive unchanged. Temporary copies contain target arrays
and BF calibration only. RawSet reads the original read-only LED/phase memmaps.
No marker fluorescence images are opened. Run after fitting diagnostics/pospos_draw0.npz.
"""
import json
from pathlib import Path
import tempfile
import numpy as np
import torch
from test_n049 import ROOT, SCRATCH, run_trainer

DATA = Path('/hnvme/workspace/v181bb10-bsccm')
with tempfile.TemporaryDirectory(dir=SCRATCH) as work:
    work = Path(work); tiny = work / 'data'; (tiny / 'pack').mkdir(parents=True)
    for name in ('led_array.npy', 'dpc.npy'):
        (tiny / 'pack' / name).symlink_to(DATA / 'pack' / name)
    with np.load(DATA / 'targets_v5.npz') as z:
        t = {k: z[k] for k in ('Y', 'mask', 'split', 'batch', 'hist_row', 'names', 'cls3', 'cls10')}
    # Limit evaluation only. All source data/provenance and original source splits remain exact.
    ev = np.flatnonzero((t['batch'] == 1) & (t['split'] == 2) & (t['hist_row'] < 0) & t['mask'][:, 0])[:24]
    t['split'][t['batch'] == 1] = 1; t['split'][ev] = 2
    np.savez(tiny / 'targets_v5.npz', **t)
    with np.load(DATA / 'cellmeta.npz') as z:
        np.savez(tiny / 'cellmeta.npz', **{k: z[k] for k in ('pos_y', 'pos_x', 'slide')})
    with np.load(DATA / 'led_norm.npz') as z:
        np.savez(tiny / 'led_norm.npz', slide_scale=z['slide_scale'][:, :1], s_c=z['s_c'][:1])
    np.save(tiny / 'backgrounds_p50.npy', np.load(DATA / 'backgrounds_p50.npy', mmap_mode='r')[:1])
    g = run_trainer(ROOT / 'scripts/train_lf.py', tiny, work / 'out',
                    ['--pospos', str(ROOT / 'diagnostics/pospos_draw0.npz'), '--pos-w', '.25'])
    assert len(g['tr']) == 8 and np.array_equal(g['ev'], ev)
    assert torch.isfinite(g['model']['pos'].weight.grad).all()
    with np.load(work / 'out/eval_preds.npz') as z:
        assert np.isfinite(z['P']).all() and np.isfinite(z['P_pos']).all()
        assert z['P_pos'].shape == (24, 8)
        result = dict(cpu=True, source_draw=0, held_out_draw=1, train_steps=2,
                      training_global_rows=g['tr'].tolist(), evaluation_global_rows=g['ev'].tolist(),
                      single_marker_supervised_entries=int(g['_zm'][g['tr']].sum()),
                      train_loss=g['log'][0]['train_loss'],
                      intensity_pearson_CD45=g['log'][0].get('r/single:CD45'),
                      probability_min=float(z['P_pos'].min()), probability_max=float(z['P_pos'].max()),
                      note='Tiny functionality smoke, not a GOAL or AUROC efficacy measurement. BF LF input only.')
    (ROOT / 'diagnostics/real_cpu_smoke.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))
