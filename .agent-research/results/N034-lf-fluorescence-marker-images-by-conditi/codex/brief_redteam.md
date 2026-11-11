# Red-team brief: N034 LF -> fluorescence-image flow matching (Claude wrote it; you review it)

Repo: ~/BSCCM (branch bsccm-data-model). READ ONLY for the repo. Write your report ONLY to
.agent-research/results/N034-lf-fluorescence-marker-images-by-conditi/codex/redteam_N034.md
Scratch / pip / caches / any temporary file: /tmp/$USER/codex-N034 only (never results/ or /hnvme). Do not run GPU jobs, do not submit
SLURM jobs, do not train. Small CPU checks (numpy on a few hundred cells via mmap) are fine; the container is
/hnvme/workspace/v181bb10-bsccm/env/bsccm.sif (apptainer exec ... python3 -I).

Files: scripts/fm_prep.py, scripts/fm_common.py, scripts/fm_train.py, scripts/fm_eval.py (they reuse scripts/vs_common.py and
bsccm/gpu_norm.py). Context: .agent-research/notebook/items/N034-*.md (Goal/Plan), .agent-research/notebook/RULES.md,
KNOWLEDGE.md (K001, K007, K019). Data: /hnvme/workspace/v181bb10-bsccm/pack/{fluor,led_array,dpc}.npy (N=412941; fluor uint16 (6,128,128)),
targets_v5.npz (Y, mask, split, batch, hist_row, slide), fm_prep.npz (written by fm_prep.py), bsccm_index.parquet.

Task: find defects that would make the N034 conclusions wrong. Specifically check:
1. Leakage: any test-blood-draw information or test cells entering training, the readout fit, or target normalisation in a way that
   inflates the 'newdraw' numbers; Giemsa cells (hist_row >= 0) anywhere in training/eval.
2. Target transform correctness (bg, gain, a_c, asinh/YS, inverse Prep.linear), the position-gain fit (is it fitted sensibly; can it
   remove real biology? cells are randomly placed), int32 handling of uint16 fluor.
3. Flow-matching correctness: interpolation/velocity target convention vs the Heun sampler direction (t=0 noise -> t=1 data), CFG null
   condition consistent between training dropout and sampling, EMA, time embedding scale, regression mode.
4. LF/fluor joint translation alignment; does LF normalisation (GPUNorm) depend on anything test-specific.
5. Evaluation validity: readout yardstick (ridge on radial integrals, fitted on real training-draw images), the r_sample / r_mean /
   r_avgread / W1 / rank-histogram calibration definitions, per-stain label mapping (LAB), masked labels, sample sizes, seeds, whether
   the comparisons sample-vs-regression are fair. Ordering assumption of the unshuffled DataLoader vs rows.
6. Resume/preemption correctness of fm_train.py (cursor, EMA, RNG, LR schedule).
Report: numbered findings, each with severity (BLOCKER / MAJOR / MINOR), file:line, a concrete failure scenario, and how you verified it
(reproduced vs reasoned). No code changes in the repo.
