# Codex WRITE brief (N093, BSCCM). Hat: write. Writable root = /home/hpc/v181bb/v181bb10/BSCCM/.agent-research/results/N093-class-conditional-giemsa-marker-matching/codex ONLY; repo read-only; scratch /tmp/n093_codex. CPU smoke only,
# no SLURM/GPU. Env: /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif (apptainer exec -B /hnvme ... python3 -I -B).
## Context (read first)
No BSCCM cell has both a Giemsa (GS) photo and a fluorescence marker measurement; GS -> marker predictions so far go through the
label-free (LF) bridge: GS students distil an LF teacher's outputs (TT6c, output distillation) or latent (hub). Read the clinical-transfer
harness: .agent-research/results/N056-e3-clinical-transfer-class-anchored-giem/codex/ (README.md, n056_eval.py, n056_contrasts.py,
n056_hub_core.py, n056_legacy.py, teachers.json, v2/), N058 (results/N058-confirm-hub-tt6c-on-mll23-fine-contrasts/: code/, n058_per_run.csv,
n058_decide.py), items .agent-research/notebook/items/N056-*, N057-*, N058-*, N008-* (PBC classifier on frozen DinoBloom features applied to
BSCCM Wright cells), N017-* (GS->MK estimate without pairs), N054-* (autofl truth on held-out GS slides). Features: /hnvme/workspace/v181bb10-bsccm/emb/
(giemsa_dinobloom_B*.npy, pbc_dinobloom_B*.npy, pbc_labels.npy, pbc_classes.json). User request: "pair Giemsa cells with marker profiles only
within the same cell type".
## ONE task: two new GS -> marker 'students' that need no GS-marker pairs, scored by the SAME harness as T / H_matched
F1 class-mean: a GS cell-type classifier (frozen DinoBloom-B features; classes = the anchored classes the harness already uses from PBC half A,
plus whatever the harness's anchor set allows; never PBC half B or MLL23 labels) gives p(class | GS image); marker prediction = sum_c p(c) *
mu_c, where mu_c = mean LF-teacher marker prediction (per teacher run t0-t2, the same teachers.json runs) over BSCCM LF cells of class c.
The class of an LF cell = the LF teacher's own class head if it has one, else a classifier trained on the BSCCM gated cls10/cls3 labels of
draw-0 pooled cells (targets npz) applied to LF features -- document which; map classes between PBC and BSCCM explicitly in a table.
F2 within-class OT: for each class c, entropic OT (POT, already used in n056_ot_align.py) between GS embeddings of class-c GS cells and
LF-teacher latents (or predicted marker vectors) of class-c LF cells; prediction = barycentric projection of the matched LF cells' teacher
marker predictions. Fit only on training-side cells (BSCCM GS cells of the training slide / PBC half A anchors as in the harness).
Score both with the harness's existing metrics and output format so they drop into n058_per_run.csv-style tables: MLL23_15, primary16,
PBC10, all25 per teacher; plus same-cell autofluorescence r on the held-out GS slide (N054 definition) and per-marker agreement r with the
LF teacher on the 15.7k paired cells (held-out slide). Same teacher runs, same eval sets, deterministic seeds.
Deliver: n093_class_match.py (+ any helper), tests/ (CPU smoke on small fixtures + one real-data dry run limited to a few hundred cells),
README.md with exact commands (CPU job with 48 cores unless a GPU clearly helps; say why), <= 300-word handback incl. the class-mapping table.
Constraints: inductive (no MLL23 / PBC half B labels, no held-out slide labels in fitting); no hand-written marker rules (classes -> marker
profiles must come from BSCCM data via the LF teacher).
