# Red-team brief: N036 (label position detrend cross-fit) + N037 (GOAL retrain on those labels) -- BSCCM repo, branch bsccm-data-model

You are the red-team reviewer. Claude wrote this code. Find real defects; for each, give a concrete reproduction (command + observed
output) or mark it [unverified]. Do NOT edit repo files. Write your report to results/N036-label-position-detrend-v7-cross-fit-the/codex/redteam_N036_N037.md.
Scratch (pip caches, temp files, venvs) ONLY under /tmp/codex_n036_rt/, never under the repo, results/ or /hnvme.
Python env: `apptainer exec --bind /hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I ...`
(login node: keep any computation light, < 5 min CPU; data at /hnvme/workspace/v181bb10-bsccm, read-only for you).

## What changed
1. scripts/make_targets_v4.py: env BSCCM_DETREND=xfit -> per slide x target, cells split into 5 FOV folds by a hash of the global FOV
   id; each cell's position trend comes from an HGB fitted on split-0 Giemsa-free cells of the OTHER folds; trend zeroed unless it
   lowers out-of-fold split-0 variance by > 1 %; then centred on split-0 clean cells. Outputs targets_v4cx.npz. (`git diff scripts/make_targets_v4.py`)
2. scripts/make_targets_v5.py: reads/writes the x-suffixed files under the same env. (`git diff scripts/make_targets_v5.py`)
3. scripts/n036_detrend_diag.py: held-out diagnostics (FOV-mean variance share per split; held-out position predictability).
4. results/N036-label-position-detrend-v7-cross-fit-the/n036_rescore.py: re-scores the GOAL-best predictions on v5/v5c/v5cx/v3c.
5. results/N037-goal-retrain-on-cross-fitted-labels-n031/n037_score.py: GOAL scorer for N031 vs N037 runs (jobs.txt has commands).
Notebook items: .agent-research/notebook/items/N036-*.md and N037-*.md (claims to check). GOAL definition: .agent-research/notebook/GOAL.md.

## Claims to attack
- C1: xfit never uses a cell's own FOV to compute its trend (check the fold logic, including the `te` mask indexing P[m][te]).
- C2: the batch-held-out GOAL test cells (eval_preds.npz G of runs n031_*) contain no split-0 cells, so GOAL test labels were never in-sample detrended.
- C3: N037 retrain differs from N031 ONLY in --targets (compare result.json args of runs/n031_{base,zoom2}_{b10,b01} vs runs/n037_*).
- C4: N037 ensemble GOAL .6537 vs N031 .6527 is computed identically to the N031 scorer (results/N031-*/score_inductive.py).
- C5: anything in make_targets_v4/v5 that silently changes the DEFAULT (hgb) path outputs vs before the edit.
Also look for: label leakage from test splits into the fit, masks/clean handling, fold degeneracy on small slides, the >0.99 keep rule
using in-sample information, np.load lazy-file issues with multiprocessing.

Report format: one section per finding: severity (blocker/major/minor), claim affected, evidence (command + output), suggested fix.
End with a verdict line per claim C1-C5: HOLDS / BROKEN / UNVERIFIED.
