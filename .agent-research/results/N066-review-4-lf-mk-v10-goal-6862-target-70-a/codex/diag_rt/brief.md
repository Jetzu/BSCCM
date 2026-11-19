# Codex RED-TEAM brief (N066 diagnosis). Hat: red-team -- change nothing in the repo; write only diag_redteam.md in this folder; scratch in /tmp.
Claim (diagnosis.md): "The remaining gap to .70 is label error from the fluorescence background model -- FOV-clustered and partly
independent between background models -- on weak/pooled targets, concentrated in b01. It acts on evaluation labels (unreachable
inductively) and on training labels (reachable: train on consensus targets over background models, evaluate on v10)."
Evidence: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N066-review-4-lf-mk-v10-goal-6862-target-70-a/diagnosis.md, /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N066-review-4-lf-mk-v10-goal-6862-target-70-a/attempts.md, scripts /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N066-review-4-lf-mk-v10-goal-6862-target-70-a/n066_perslide.py, n066_fovresid.py, n066_altlabels.py and their CSVs;
N064 reconstruction: /hnvme/workspace/v181bb10-bsccm/runs/n064_ceiling/ (half_labels.npz: G, A, B, full, names) and its report.md;
labels /hnvme/workspace/v181bb10-bsccm/targets_v5cx_v10.npz; predictions /hnvme/workspace/v181bb10-bsccm/runs/<run>/eval_preds.npz.
Python env: apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I ...
Do not fix anything. Find why this claim is wrong: (a) what trivial explanation gives the consensus gain .6929 -> .6996 (e.g. averaging
ANY second noisy variable correlated with y raises r; shrinkage of the bright tail; local-bg labels being a function of the same pixels
the model predicts)? Construct the control that separates "independent background error" from those (e.g. consensus with a shuffled
or noise-matched second label). (b) Is the FOV-share excess an artefact of cell-type composition per FOV (residual of a cell-type-
dependent bias clusters by FOV)? (c) Is the replicate-slide grp:CD45 gap (.881 vs .578) explained by label variance/range differences
rather than error? (d) Do the training-label consensus expectations follow? Reproduce numbers, give each finding a severity, and say
NOTHING BREAKS if none. Keep diag_redteam.md <= 2k tokens with every number traceable to a command you ran.
