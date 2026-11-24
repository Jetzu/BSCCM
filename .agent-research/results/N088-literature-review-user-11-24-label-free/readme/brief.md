# Codex FIX ROUND brief (BSCCM README). Hat: fix (writer). Writable root = /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N088-literature-review-user-11-24-label-free/readme ONLY; repo read-only; scratch /tmp/readme_codex2.
Edit /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N088-literature-review-user-11-24-label-free/readme/README.md (your round-1 draft) to fix EVERY item in /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N088-literature-review-user-11-24-label-free/readme/factcheck_claude.md (13 errors, 8 overclaims, the command/usability
problems). Claude verified the key ones first-hand (N084 per-marker CD16 .900 / CD19 .360 / HLA-DR .480 / CD56 .531 / CD123 .596 from
figures/overview/data/per_target.csv rows 'Ensemble N084'; N007 verdict worse; within-slide retrieval caveat from CAPTIONS.md).
Also update to the current state (read: .agent-research/results/N088-literature-review-user-11-24-label-free/synthesis.md and lit_A/B/C_*.md;
.agent-research/notebook/items/N087-*.md, N089-*.md, N090-*.md; figures/overview/CAPTIONS.md):
- N087: list its arms and jobs; the conditional flow-matching decoder is DONE: on frozen features it does not restore co-expression
  (sandbox mean of 3 donors: per-marker r linear head .588 / MLP .549 / flow mean .554 / flow one sample .484; CD4-CD8 in T cells
  true -.804, predicted -.36 / -.27 / -.22 / -.31; spread (sd ratio) .74 / .82 / .82 / .97); BSCCM GOAL on frozen features: model head
  .6730, flow mean .6507, MLP .6369 (flow-vs-MLP confounded by capacity/update count, Codex red-team) -> not a GOAL route.
- N088: literature review done (3 reviews, 104 references) with the synthesis portfolio; N089: GitHub + Hugging Face release
  (private HF repo Jetzu77/BSCCM holds weights-only checkpoints: list the folders); N090: gentle fine-tuning launched (pre-registered bar .6760).
- Do NOT link to files that are untracked in git (check with git -C /home/hpc/v181bb/v181bb10/BSCCM ls-files); link INDEX.md rows instead.
- Rebuild commands must write to NEW output directories (e.g. runs/rebuild_<name>) and say so in the code blocks; state that
  scripts/gpu_job.sbatch depends on a cluster helper not in the repo (give the plain apptainer command as the portable alternative);
  name the Python requirement (any Python with numpy + pandas) instead of a personal conda env; no 'exit 1' in interactive snippets.
- Keep everything else; keep the no-e-mail / no-mentor-names rule. Add a short 'Data and checkpoints' section (data not in the repo;
  checkpoints on the private Hugging Face repo Jetzu77/BSCCM).
Deliver the edited /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N088-literature-review-user-11-24-label-free/readme/README.md and a <= 150-word handback listing what changed.
