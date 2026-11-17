# Codex WRITER brief: N045 -- SOTAlign (semi-supervised OT alignment) for Giemsa <-> label-free cells in BSCCM

Repo /home/hpc/v181bb/v181bb10/BSCCM (read-only for you; deliver code into your out-dir only). SOTAlign source (MIT, cloned, read-only):
/hnvme/workspace/v181bb10-bsccm/external/SOTAlign (paper arXiv 2602.23353, ICML 2026: CCA teacher fitted on the few pairs; alignment
layers f,g trained with a SigLIP/SAIL loss on pairs + KLOT regulariser = KL between entropic OT plans of teacher and student affinities
on UNPAIRED batches; core in src/optimal_transport/{matching.py,similarity.py}, src/models/{sail_model.py,loss.py,linear.py}).
Data (read-only): /hnvme/workspace/v181bb10-bsccm. Scratch ONLY under /tmp/codex_n045/ (uv venv: torch CPU, numpy, scipy, sklearn,
POT if needed). apptainer does not work in your sandbox. No GPU here: deliver code + a CPU smoke on <= 2k samples; I run the GPU job.
Read first: .agent-research/notebook/KNOWLEDGE.md (K001 premise: no cell has Giemsa + markers; K005, K006, K011, K015, K016, K021-K025),
items N009 (binding v0: Procrustes vs linear SigLIP adapter on frozen features, cross-slide R@10 .018 vs .033;
.agent-research/results/N009-stage-2-binding-v0-on-frozen-sup-lf-feat/), N021 (class-anchored alignment to PBC), N032 (clinical
contrast AUROCs, transfer_all.csv). Rules: fluorescence never an LF input; Giemsa cells never in LF-teacher training; labels only from
targets files; report everything, tune nothing on test slides.

Embeddings available:
- Giemsa (BSCCM, 15,712 cells on 2 slides, one per blood draw): emb/giemsa_dinobloom_B.npy (+ _scalematched variant, see emb/scale.json);
  find the row -> global index mapping (pack/hist_cells.npy, targets_v5c.npz hist_row, runs/n029_r18_v4c_fov/probe/lf_feats_hist_*.npy).
- Label-free: runs/n029_r18_v4c_fov/probe/lf_feats_all.npy (412,941 x 512, supervised r18 marker teacher, FOV split) and lf_feats_hist*.npy
  for the Giemsa cells' own LF crops; alternative SSL features runs/n007_ssl_aug/probe_fov/lf_feats_all.npy.
- Unpaired clinical Giemsa: emb/pbc_dinobloom_B.npy + pbc_labels.npy + pbc_classes.json (PBC, labelled cell types for EVALUATION only).
  MLL23 images are in external/mll23 without precomputed features (optional; skip unless trivial).

Build scripts/n045_sotalign.py (deliver in out-dir) that reuses SOTAlign's MatchingModel / CCA / AlignmentLayer on numpy embeddings:
X = Giemsa features (DinoBloom-B), Y = LF features. Pairs = BSCCM Giemsa<->LF same cells. Unpaired X = PBC (+ BSCCM Giemsa of the
held-out slide is NOT allowed as unpaired when that slide is the test slide); unpaired Y = random non-Giemsa LF cells (e.g. 100k).
Protocols: cross-slide (train pairs on slide 1, test on slide 2, and reverse) and FOV split within slide. Arms (same seeds, same
budget): (1) CCA only (teacher), (2) SigLIP pairs only (no unpaired; ~ N009 adapter), (3) SOTAlign (pairs + KLOT with unpaired PBC
+ LF), (4) SOTAlign with LF-unpaired only. Hyper-parameters from scripts/sotalign.sh defaults scaled to our sizes (document choices;
choose only on the training slide's held-out FOVs).
Metrics (write CSV/JSON): (a) test-slide exact-cell retrieval Giemsa->LF R@1/R@10 (chance = 10/n); (b) marker head transfer: ridge
head from aligned LF space to the curated labels (targets_v5c, non-Giemsa labelled cells, training draw only), applied to aligned
Giemsa features of test-slide cells -> Pearson r with the Giemsa cells' own measured autofluorescence (autofl truth; the only same-cell
molecular truth on Giemsa cells; see N012/N024 for the detrended truth definition) vs the distillation student numbers in KNOWLEDGE;
(c) PBC: kNN accuracy of PBC cell types in the aligned space (sanity) and the N021/N032 lineage contrasts (AUROC of predicted
markers between PBC classes, e.g. neutrophil>lymphocyte CD16, monocyte>rest CD14, lymphocyte>neutrophil CD3). Include the raw-feature
and N009-style baselines in the same table.
Deliver: code, a CPU smoke (<= 2k pairs, few steps) with its output, a GPU run script (single H100, preempt-safe or < 30 min), README.md
with exact commands, and report.md (design decisions, mapping checks, risks). Do not edit repo files.
