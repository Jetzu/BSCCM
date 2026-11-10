# N033 · Virtual-stain data engine: Giemsa student trained additionally on virtual Wright images of 318k marker-labelled cells with MEASURED labels (TT6c + synthetic, w 1.0), paired vs N029 TT6c (same teachers, seeds)

type: exp · date: 2026-11-10 · status: open
goal_effect: pending · scale: module

## Goal
The Giemsa student so far learns markers only from LF-teacher OUTPUTS on 15.7k paired cells. With virtual staining (N028) we can render Giemsa-like images for the ~318k marker-labelled cells and supervise the student with MEASURED fluorescence labels. Does that improve clinical transfer (MLL23/PBC) and same-cell autofl truth?

## Plan
Generator = N028 FOV model (trained on split-0 GS pairs only). Render all 317,935 non-GS split-0 cells (scripts/vs_generate.py -> /hnvme/workspace/v181bb10-bsccm/runs/n033_synth_fov/images.npy, 15.6 GB uint8). Labels = targets_tt6c.npz (v4c markers + v6c unstained-only autofl; scripts/make_targets_tt6c.py). Student: TT6c (N029 teachers t0) + masked MSE on measured labels of a synthetic batch per step (w 1.0), virtual 128 px image embedded at the registration field (.67 x 398 px) in a border-median canvas, student units x rgb_scale/scale. 3 seeds, paired vs N029 tt6c_s0-2 (same teachers, seeds, steps). Eval: within-slide autofl truth (v4c clean), MLL23 fine mean, PBC-B. Caveat: the generator only sees LF, so synthetic images cannot carry information beyond LF; the hope is better DOMAIN coverage (318k cells) for Giemsa->marker mapping. Jobs .agent-research/results/N033-virtual-stain-data-engine-giemsa-student/jobs.txt. ~2.5 GPU-h.

## Changes
- scripts/vs_generate.py (render virtual stains for any LF rows), scripts/make_targets_tt6c.py, scripts/train_gs_bind.py --synth/--synth-targets/--w-synth (SynthSet + l_syn).

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N033 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N033-slug/ + external absolute paths -->

## Next
