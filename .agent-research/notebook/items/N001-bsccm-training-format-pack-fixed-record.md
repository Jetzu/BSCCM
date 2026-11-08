# N001 · BSCCM training-format pack (fixed-record npy memmaps) + dataset audit

type: task · date: 2026-11-08 · status: done
goal_effect: n/a · scale: ? · value: - · why: infra

## Goal

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Dryad v13 (2025-05-06) full BSCCM, 50 chunks sha256-verified (scripts/fetch.py), streamed into BSCCM.zarr.zip (215 GB, 13.7 M entries, ZIP_STORED).
- build_index.py: 1 .zarray variant per modality, 0 missing chunks (9.50 M led, 2.48 M fluor, 0.41 M dpc, 15,712 x 4 histology).
- pack.py job 1274096 (preempt_cpu, 48 cores, 8.5 min): 1,860 blocks, 0 missing chunks.
- verify_pack.py (independent decode by name via zipfile+numcodecs, no offset index): 0 mismatching arrays over 300 cells x 30 arrays + 60 histology; 0 all-zero BF rows over all 412,941 cells; 0 all-zero sampled histology rows.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N001 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
- /hnvme/workspace/v181bb10-bsccm/pack/{led_array (N,23,128,128) u2, fluor (N,6,128,128) u2, dpc (N,128,128) f4, histology (15712,398,398,3) u2, hist_cells}.npy (np.load(mmap_mode='r'))
- targets.npz (v1: asinh(x/50), masks, gates, FOV split — to be revised per N002 data corrections), led_stats.npz (provisional log1p z-score), bsccm_index.parquet, bsccm_sm.parquet
- results/N001-bsccm-training-format-pack-fixed-record/audit.txt, targets_build.txt, verify_pack.txt, led_stats.txt
<!-- files in .agent-research/results/N001-slug/ + external absolute paths -->

## Next
