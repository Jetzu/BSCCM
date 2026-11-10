# Red-team brief (adversarial reviewer; read-only analysis; do not edit repo files)

Repo /home/hpc/v181bb/v181bb10/BSCCM (branch bsccm-data-model). Write ONLY into your cwd:
/home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N025-giemsa-chain-without-the-trade-off-two-t/codex/
Scratch/pip/caches: /tmp/$USER/codex-N025 only (mkdir; delete before finishing). Never write to /hnvme or elsewhere in results/.
Python: apptainer may fail on this node (fuse); if so use /home/hpc/v181bb/v181bb10/miniforge3/envs/triad/bin/python (numpy/pandas/sklearn). Light CPU only, < 10 min per computation, single-threaded BLAS.

## Context
BSCCM: label-free (LF) LED images pivot between Giemsa/Wright (GS) cell images and surface markers. A GS "student" (DinoBloom-B fine-tuned) is distilled from an LF "teacher" (ridge heads on frozen LF features) on 15.7k GS<->LF paired cells; markers are never measured on GS cells, but autofluorescence (autofl) IS measured on the same unstained cells -> the only same-cell GS truth (eval only). Read the item first: .agent-research/notebook/items/N025-giemsa-chain-without-the-trade-off-two-t.md (and N024 for the trade-off).
Claims of N025:
 C1. Two-teacher distillation (markers from the v4 r18 teacher, autofl from a v5/v6 r18 teacher; scripts/train_gs_bind.py --aux-feats/--aux-targets/--aux-names, diff in codex/train_gs_bind.diff) removes the N024 trade-off: within-slide GS autofl truth r tt6 .578/.855 vs v4 .383/.689, while MLL23 clinical fine-contrast mean stays .611 vs v4 .622 (v5/v6 single teachers .536/.527).
 C2. v6 labels (scripts/make_targets_v6.py: autofl only on unstained slides) give a cleaner autofl target (within-slide b0 .579 vs v5 .510) and leave the LF->MK GOAL flat (.639 vs .6404).
 C3. Within-slide Pearson is the right metric (pooled two-slide r is not comparable across label versions) and nothing leaks GS autofl truth into training.
Files: scripts/train_gs_bind.py (+diff), scripts/make_targets_v6.py, scripts/gs_autofl_within.py, scripts/aggregate_transfer.py, scripts/apply_gs_student.py, results in the N025 dir (n025_metrics.csv, transfer_multiseed.csv, goal_runs_v6.json), runs in /hnvme/workspace/v181bb10-bsccm/runs/n025_* (result.json, test_preds.npz), targets_v{4,5,6}.npz in /hnvme/workspace/v181bb10-bsccm/.

## Your ONE task
Try to break C1-C3. In particular: (a) LEAKAGE: can GS-cell autofl truth reach training through the aux teacher (ridge rows must exclude hist_row>=0 GS cells; LF teacher training excluded GS cells via include_gs=0 -- verify from runs/n025_r18_v6_fov/result.json and train_lf.py), or through v6 normalisation statistics (split-0 rows include GS cells; affine only?), or through probe features; (b) is the within-slide r computed on the right rows/slides and is it really invariant to the v4/v5/v6 per-slide transforms; (c) is the comparison fair (same steps/anchors/seeds/holdout; reference arms built from older runs - check args in result.json); (d) bugs in the aux code path (index mixing between names of the two targets files, normalisation with stained_tr from the primary targets, teach overwrite order, bound loop skip); (e) MLL23 metric: 12 pre-registered fine contrasts, seed spread (n=3) vs the claimed equivalence tt6 .611 vs v4 .622; (f) whether v6 'cleaner' could instead be fewer/different cells or a different scale; (g) anything that makes "two-teacher removes the trade-off" misleading.
Reproduce numbers yourself where possible; cite file:line or the exact command for every finding.

## Contract
Write codex/redteam_N025.md: numbered findings with severity (blocker/major/minor), evidence (command + output), concrete fix. Final message <= 500 words. Modify nothing outside codex/. Delete /tmp/$USER/codex-N025 before finishing.
