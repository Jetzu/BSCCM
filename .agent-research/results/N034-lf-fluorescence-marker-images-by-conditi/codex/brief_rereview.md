# Re-review brief: N034 fixes after your first red-team (codex/redteam_N034.md)

Repo ~/BSCCM, READ ONLY. Write ONLY .agent-research/results/N034-lf-fluorescence-marker-images-by-conditi/codex/rereview_N034.md.
Scratch/pip/caches: /tmp/$USER/codex-N034b only. No GPU, no SLURM, no training. Small numpy checks on stored arrays are fine
(container may not mount; you previously used host python with numpy installed under /tmp).

Claude changed scripts/fm_prep.py (gain model), scripts/fm_train.py, scripts/fm_eval.py (rewritten), scripts/fm_figure.py (new) and
added scripts/n034_setmatch_sim.py (scalar set-level matching bench). Item: .agent-research/notebook/items/N034-*.md (Evidence lists the
claimed fix for each of your findings F1-F8, the results and the sampler sweep). Results: results/N034-*/n034_eval_v2.csv,
n034_sampler_sweep.csv, setmatch/setmatch_sim.csv; eval outputs /hnvme/workspace/v181bb10-bsccm/runs/n034/eval_*/eval.json + per_cell.npy.

Tasks: (1) for each of F1-F8 say RESOLVED / PARTIAL / OPEN with file:line evidence; (2) find NEW defects in the changed code that would
change the conclusions in the item (especially fm_eval.py: conditional W1 strata, CRPS, shuffled control construction, seeds, split-2
restriction, readout/template); (3) check n034_setmatch_sim.py: is the set-level learner and its comparison fair (paired vs same vs other;
bag construction; does 'same' at n=1 equal paired; can 'other' leak pairing); (4) are the item's claimed numbers consistent with the CSVs.
Severity BLOCKER/MAJOR/MINOR, concrete failure scenario, reproduced vs reasoned. No repo changes.
