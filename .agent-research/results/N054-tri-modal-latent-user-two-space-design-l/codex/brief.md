# Codex WRITER brief: N054 -- tri-modal latent spaces (label-free LF, fluorescence markers MK, Giemsa GS) in BSCCM

Repo /home/hpc/v181bb/v181bb10/BSCCM (READ-ONLY for you; deliver everything into your cwd = this codex/ folder). Data (read-only):
/hnvme/workspace/v181bb10-bsccm. Scratch ONLY under /tmp/codex_n054/ (uv venv: torch CPU, numpy, scipy, sklearn, timm==1.0.31, pandas,
pillow). apptainer does not work in your sandbox and there is no GPU: deliver code + a CPU smoke test on tiny subsets; I run the GPU jobs.

## Context (read first)
- .agent-research/notebook/KNOWLEDGE.md (K001: NO cell has both Giemsa and markers; GS<->LF are the SAME cells (15,712, two unstained
  slides, one per blood draw); LF<->MK are the same cells (single-marker / pooled-panel cells); K011, K022, K025, K035 = current Giemsa
  chain numbers), item N052 (the plan this implements), N045 (SOTAlign: aligned spaces find cells but carry little marker info).
- Existing code to reuse (copy + adapt, do not import from scripts/ by path hacks unless clean): scripts/train_gs_bind.py (image-level
  DinoBloom-B GS encoder, freeze first 8 of 12 blocks, GPU stain/optics augmentation, SigLIP + distillation, two-teacher aux
  features, cross-slide --holdout batch1|batch0, eval: GS->LF retrieval, autofl truth r, student-vs-teacher r per marker);
  scripts/train_bind.py (N009: marker SET encoder = marker-id token + value MLP + small transformer, SigLIP with near-identical
  marker readouts not used as negatives); scripts/apply_gs_student.py (PBC lineage AUROCs + MLL23 fine contrasts, pre-registered).
- Baseline to beat = TT6c output-distilled student: runs/n029_tt6c_x_batch1 and runs/n029_tt6c_x_batch0 (gs_bind.pt, result.json,
  args inside: LF teacher features runs/n029_r18_v4c_fov/probe (markers, targets_v4c.npz) + aux runs/n029_r18_v6c_fov/probe (autofl,
  targets_v6c.npz), 2500 steps, bs 128, lr 1e-4, freeze 8). Cross-slide autofl truth r student .715 (holdout batch1).

## The question
The user proposes TWO latent spaces aligned by flow matching: space 1 = LF encoder + marker-matrix encoder (an MLP that encodes the
cell's row of the cells x markers matrix, missing entries masked); space 2 = LF encoder + Giemsa encoder (DinoBloom-B); then align
space 2 -> space 1 (flow matching or similar) and decode markers from space 1. The alternative (Claude) is ONE LF-anchored hub: the GS
encoder regresses the LF latent of the same cell (latent distillation, + optional SigLIP), markers decoded by one head on the hub.
The user also wants to test "hidden information": a cell labelled only for CD19 whose embedding lies near CD45-bright cells should be
decodable as CD19+ AND CD45+ -- i.e. marker completion from the shared space, and whether conditioning on the observed marker values
(marker-matrix encoder input) adds information beyond the LF image.

## Build (one package of scripts in your folder; GPU-ready, preempt-safe = resume from <out>/ckpt.pt, save on SIGTERM/USR1)
Fixed for ALL arms: LF features = frozen per-cell features from runs/n029_r18_v4c_fov/probe/lf_feats_all.npy (512-d) concatenated with
runs/n029_r18_v6c_fov/probe/lf_feats_all.npy (512-d) (find the Giemsa cells' own LF rows via lf_feats_hist*.npy / hist_row as
train_gs_bind.py does); labels = targets_tt6c.npz (13 targets: 8 single markers, 4 pooled groups, autofl; mask = observed);
GS encoder = DinoBloom-B image-level exactly as train_gs_bind.py (same aug, freeze 8, bs 128, lr 1e-4, same step count 2500);
cross-slide protocol: --holdout batch1 (train on GS slide of draw 1, test on the draw-2 GS slide) and batch0 (reverse); LF-side models
are trained on non-Giemsa LF cells of FOV split 0 (both draws, as the teachers), validated on split 1, never on Giemsa cells.

Arms:
- A1 two-space + PAIRED bridge flow (user's design, strong version). Space 1: P1 (MLP on LF feats) and E_M (marker-matrix MLP: input
  [y*mask, mask] for the 13 targets, 2x512 hidden, d=256) trained with SigLIP(P1(f), E_M(row)) on labelled LF cells (negatives with
  near-identical observed readouts ignored, as train_bind.py) + decoder D_M: space-1 vector -> 13 targets (masked MSE on both P1(f) and
  E_M(row) inputs, the latter with the decoded entry masked out of the input = no identity shortcut). Space 2: P2 (MLP on LF feats) +
  GS encoder, SigLIP on GS<->LF pairs of the training slide. Alignment: conditional flow matching / stochastic-interpolant BRIDGE from
  z2 to z1 trained on LF cells, where every LF cell gives the pair (P2(f), P1(f)) (frozen P1, P2); small MLP velocity net with time
  embedding, sigma_bridge in {0, 0.1}; inference on a Giemsa cell: z_G = E_G(img) -> integrate 32 Euler steps -> z1 -> D_M -> markers.
- A2 two-space + UNPAIRED alignment (user's design, literal version): same spaces; flow between the MARGINALS only (minibatch-OT
  conditional flow matching, source = z2 of LF cells, target = z1 = E_M(row) embeddings of marker rows from DIFFERENT cells; no
  correspondences). Same inference path.
- A3 LF-anchored hub (Claude). One space: P_L (MLP on LF feats) -> z (d=256) with marker head H (masked MSE, 13 targets) on labelled
  LF cells (+ the same E_M SigLIP term so the hub also carries the marker matrix). GS encoder E_G trained on pairs to REGRESS the
  frozen hub latent of its own LF cell (MSE on L2-normalised z + w_sig * SigLIP(E_G, P_L), w_sig in {0, 0.5}). Markers on GS = H(E_G(img)).
- A0 baseline: re-evaluate the existing TT6c checkpoints with YOUR eval code (identical metrics), do not retrain.
Completion tests (LF side only, no GS; truth exists):
- C1 autofl completion on single-marker cells (they have measured autofl): predict autofl from (a) LF only, (b) LF + the observed
  marker entry via the marker-matrix encoder (autofl masked out of the input), fused model = MLP over [P_L(f), E_M(masked row)] trained
  with random input masking (mask each observed entry with p=.5, predict all observed entries). Eval r on held-out FOVs (split 2) AND
  held-out draw (train draw 1 -> test draw 2 and reverse; the fused model trained on one draw only for that test).
- C2 pooled-panel leave-one-group-out: the 37k pooled cells have 4 group labels (+ grp CD123/HLA-DR/CD14 failing QC in draw 2 = masked
  in the targets file). Predict each held-out group from (a) LF only vs (b) LF + the other observed groups. Same splits.
- C3 coherence (no truth, report only): among single-marker cells observed CD19-high (top 5 % of observed CD19), CD3-high, CD14-high,
  CD56-high, mean decoded value of every OTHER marker under (a) LF only and (b) fused; tabulate vs the population mean.
Metrics (CSV + JSON per arm x direction; everything measured is saved):
- M1 autofl truth r on the held-out GS slide (decoded autofl vs the GS cells' own measured autofl; detrended labels from targets_tt6c).
- M2 per-marker r(decoded GS markers, LF-teacher markers of the same cells) on the held-out slide; teacher = H/D_M applied to the cells'
  own LF features (same for all arms: use one fixed ridge teacher fitted on LF labelled training cells for this metric, and also each
  arm's own LF path) + r_est = r(student,teacher) x r(teacher, truth on held-out stained cells of that draw) as gs_mk_estimates.py.
- M3 GS->LF exact-cell retrieval R@1/R@10 on the held-out slide (chance 10/n) in the arm's matching space.
- M4 geometry: linear CKA and mutual-kNN overlap (k=10) between z_G and the matching LF latent of the same held-out cells; for A1/A2
  also between space 1 and space 2 on LF cells.
- M5 LF->MK r per target on held-out stained cells (split 2) through each arm's LF path (D_M(P1(f)) vs H(P_L(f)) vs ridge on f):
  shows what a contrastive space keeps vs a regression head.
- M6 clinical: write the GS predictions for PBC (external/pbc zip) and MLL23 (external/mll23) through apply_gs_student.py's
  pre-registered contrasts (reuse its contrast definitions and image preprocessing verbatim; make a variant that loads any arm's GS
  encoder + decoder) -> AUROC table.
Deliver: n054_trimodal.py (stages: lf_spaces | gs_encoder | flow | eval | completion; argparse; deterministic seeds; one --out dir per
arm x direction), n054_clinical.py, n054_gpu.sbatch template (partition preempt, 1 GPU, 16 cores, apptainer exec --nv --bind
/hnvme/workspace/v181bb10-bsccm /hnvme/workspace/v181bb10-bsccm/env/bsccm.sif python3 -I ...; outputs to
/hnvme/workspace/v181bb10-bsccm/runs/n054_*), README.md with the EXACT command list in dependency order and a GPU-hour estimate,
report.md. CPU smoke: run every stage end-to-end on <= 2k LF cells, <= 300 GS cells, 20 steps, tiny GS backbone stand-in (e.g. timm
vit_tiny with random init) behind a --smoke flag; record the smoke log. Rules: fluorescence never an LF input; Giemsa cells never in
LF-side training; test-slide GS cells never in any training/selection; choose hyper-parameters only on the training slide's held-out
FOVs; no new labels; do not tune on PBC/MLL23. If something in this brief is wrong about the data, say so in report.md and adapt.
Handback (<= 2k tokens): files, smoke results, deviations, open risks.
