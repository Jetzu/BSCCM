# Codex WRITE brief (BSCCM README). Hat: write. Writable root = /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N088-literature-review-user-11-24-label-free/readme ONLY; repo read-only; scratch /tmp/readme_codex.
## ONE task: write README.md (repo root text, delivered as /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N088-literature-review-user-11-24-label-free/readme/README.md) for the private GitHub repo Jetzu/BSCCM.
Audience: an ML researcher or student joining the project who knows deep learning but NOT immunology/microscopy. The user asked for
"a nice readme that explains everything, every single step from the start of what we have done to what the goal is".
Sources (read ALL, numbers must come from them, never invent): .agent-research/notebook/INDEX_ARCHIVE.md + INDEX.md (one row per item
N001-N088, status, result, hash), .agent-research/notebook/KNOWLEDGE.md (findings Kxxx), .agent-research/notebook/RULES.md if present,
.agent-research/notebook/items/N0xx-*.md (read the item when a row is unclear), figures/overview/CAPTIONS.md (figures 1-9 with numbers;
this is the most recent and red-teamed summary -- prefer its numbers where they overlap), figures/overview/*.png names, scripts/ (one line
per main script: train_lf.py, train_ssl_lf.py, train_ifc_sandbox.py, pack_sbiad452.py, make_targets_v*.py, score_goal_arms.py, fm_*.py, ...),
bsccm/ (modules), env/ (container), tests/. Repo root: /home/hpc/v181bb/v181bb10/BSCCM.
Required sections:
1. One-paragraph summary: what the project is, the goal (GOAL metric, target .70, current best numbers), the main findings.
2. Background in ML terms: the BSCCM dataset (LED-array microscope, 24-channel input, single-stain vs pooled-panel slides, two blood
   draws, Giemsa photos of 15.7k of the same cells, no Giemsa-marker pairs), external data (PBC, MLL23 clinical smears; S-BIAD452
   ImageStream with 9 markers per cell), and why the problem is hard (one marker per cell, label noise at field-of-view level, draw shift).
3. Goal and evaluation protocol: GOAL definition (common-11 Pearson r, train one draw / test the other, both directions), MM3, noise,
   pre-registration and the "improved = beat best by > 2x noise" rule, label versions v4 -> v5 -> v10 and why re-bases are not progress.
4. Timeline: EVERY item N001-N088 in order, grouped into phases (e.g. setup/audit, first models + SSL, label QC, Giemsa chain + clinical
   transfer, flow-matching/virtual staining, label re-basing v10, encoders + ensembles, heads/losses, multi-marker + external data,
   alignment, reviews, figures, current work). For each item one or two lines: what was tried, the result number, verdict. Mark reviews.
5. What worked best per part (table: LF->markers, joint decoding, LF<->Giemsa, Giemsa->clinical, generative, unpaired alignment,
   draw shift) and what did not work (with numbers).
6. Current best model and how to reproduce it (architecture: 4-member ensemble, encoders, SSL pretraining, fine-tuning recipe; the exact
   commands from the items' jobs.txt for the record N053 ensemble and its members; scoring with scripts/score_goal_arms.py).
7. Repository layout (directories, main scripts, the .agent-research notebook: items, INDEX, KNOWLEDGE, results/N###, plots), data
   locations (paths on the cluster; data are NOT in the repo), environment (apptainer container), how jobs are run (SLURM preempt queue).
8. Figures: list figures/overview/fig1-fig9 with one-line descriptions (embed with relative markdown image links).
9. Open problems and next steps (N085 running, N087 decoder/autoencoder/flow arms, N088 literature review, ideas in notebook/IDEAS.md).
Style: clear headings, tables where they carry structure, no hype, numbers with 3 decimals as in the sources, ML vocabulary explained
once. No personal data beyond what the sources contain: do NOT include e-mail addresses, and do not name supervisors/mentors (refer to
"the supervising lab"). No secrets. Length: as long as needed to cover every item (expect 600-1000 lines of markdown).
Deliver: /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N088-literature-review-user-11-24-label-free/readme/README.md and a <= 200-word handback listing any source inconsistencies you noticed (cite file:line).
