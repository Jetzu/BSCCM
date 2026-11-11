# Narrow check (round 3) of the fixes made after codex/rereview_N034.md

Repo ~/BSCCM READ ONLY; write ONLY .agent-research/results/N034-lf-fluorescence-marker-images-by-conditi/codex/check3_N034.md.
Scratch/pip: /tmp/$USER/codex-N034c only (you may reuse /tmp/$USER/codex-N034b/python numpy). No GPU/SLURM/training/inference.
Scope: (1) scripts/n034_setmatch_sim.py v2 (bag designs paired/paired_A/same_fov/same_rand/same_A/other/other_fov; learners bagmean,
ot, ot_init; v5c; split-2 tests) -- correctness, leakage (does any 'other*' arm see same-cell pairs?), budget/locality matching, and
whether results/N034-*/setmatch/setmatch_sim_v2.csv supports the item's 'B v2' Evidence + Outcome text and KNOWLEDGE K026 exactly as
written; (2) scripts/fm_eval.py hardening (derangement loop, rank quintiles, LF hash assert, saved readout metadata, --newdraw-split)
and scripts/fm_figure.py labels -- any bug introduced? (3) the N034 item's corrected claims (DEVELOPMENT labelling, calibration
failure, withdrawn registration claim) vs your rereview -- anything still overstated? Also glance at the N035 item's pre-registration
(.agent-research/notebook/items/N035-*.md): is the primary cohort (draw 1 split 1) a sound untouched test? Report numbered findings
BLOCKER/MAJOR/MINOR with file:line and reproduced vs reasoned. Keep it short.
