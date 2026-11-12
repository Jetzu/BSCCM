# Diagnosis red-team brief (N038 project review, BSCCM)
Context files (read first): .agent-research/notebook/GOAL.md, .agent-research/notebook/KNOWLEDGE.md, .agent-research/results/N038-review-lf-mk-common-11-batch-held-out-go/attempts.md, .agent-research/results/N038-review-lf-mk-common-11-batch-held-out-go/declined.md (binding constraints). Repo /home/hpc/v181bb/v181bb10/BSCCM (code scripts/train_lf.py = GOAL predictor; src/bsccm/). Data /hnvme/workspace/v181bb10-bsccm (read-only for you). Python: apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I ... ; keep any computation light (< 5 min CPU, login node). Scratch (pip, caches, temp) ONLY under /tmp/codex_n038/, never in the repo, results/ (except your report) or /hnvme. Do not edit repo files. Do not submit jobs.
Claim (written by Claude): .agent-research/results/N038-review-lf-mk-common-11-batch-held-out-go/diagnosis.md -- "the GOAL is bounded by per-cell information about CD19/CD56/CD123/HLA-DR (image- or
label-limited); the draw shift costs only ~.016 on identical cells, so the target .67 sits at the in-distribution ceiling of the
current predictor". Evidence scripts/CSVs: .agent-research/results/N038-review-lf-mk-common-11-batch-held-out-go/d1_fov_vs_batch.py/.csv, .agent-research/results/N038-review-lf-mk-common-11-batch-held-out-go/d1b_same_arch.py/.csv (runs/<run>/eval_preds.npz: P, G).
Do NOT fix anything. Questions: What simpler explanation fits the attempts equally well? Where does the D1b ceiling test mislead
(n015_sslinit_fov was trained on v4 labels, both draws, 318k cells, transductive SSL encoder n007; n023_v5 single models; different
label versions for grp targets; split-2 cells only)? Is 'in-distribution .658' really a ceiling or an artefact of the comparison?
Are the per-target conclusions (four dim targets) robust? Reproduce any finding with a command + output; otherwise mark [unverified].
Also check the scale labels in attempts.md (knob/module/paradigm). If nothing breaks, say NOTHING BREAKS.
Write the report to .agent-research/results/N038-review-lf-mk-common-11-batch-held-out-go/codex/diag_rt/diag_redteam.md.
