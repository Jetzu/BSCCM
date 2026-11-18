# N053 · LF encoder swap on the v10 GOAL: DINOv3 ViT-S/B, DinoBloom-B and Cytoland encoders fine-tuned with a learned per-LED gate + 24-channel patch embedding vs the SSL ResNet-50

type: exp · date: 2026-11-18 · status: done
goal_effect: improved · scale: module · value: 0.6862

## Goal
User ruling N052: test new LF image encoders. Does a large pretrained ViT (DINOv3 natural images; DinoBloom blood smears) or a
label-free microscopy encoder (Cytoland VSCyto2D) beat the SSL-init ResNet-50 on the v10 GOAL (LF->MK common-11, batch-held-out r)?

## Plan
Hypothesis: pretrained ViT features transfer better across draws (less draw-specific readout). Expected: modest (labels limit,
K028-K030; N050 ConvNeXt errors correlate .94-.96 with ResNet). Adapter (bsccm/encoders.py): learned per-LED gate (24 scalars,
init 1) -> resize 128 -> 256 (ViT/16) or 224 (ViT/14) = 16x16 tokens -> RGB patch embedding adapted to 24 channels (timm
adapt_input_conv) -> pooled features -> same heads/losses as the GOAL recipe. Encoder lr x0.1 (--enc-lr-mult), drop-path .1.
Arms (v10 labels targets_v5cx_v10.npz, full view, 10 epochs, lr 5e-4 heads, bs 256, seed 0, both directions):
dinov3_s (23.7 M), dinov3_b (89.8 M), dinobloom_b (88.9 M), cytoland (if weights obtainable).
Baseline = single SSL r50 base view on v10 labels (n051_v10_base_b10/_b01) and the GOAL best .6804 (2-model ensemble N048).
Success: single-model GOAL > single r50 base by > 2 x .0027; then ensemble with the r50 family (error decorrelation, cf. N050).
Kill: all ViT arms <= r50 base.
Placement: GPU (1 x H100, 16 cores) -- ViT-B fine-tuning at 256 tokens is far > 2x a 48-core CPU run.
Budget: smoke timing first (1376765/6, 30k cells, 1 epoch), then <= 4 arms x 2 directions.

## Changes

## Evidence
- Baseline single models on v10 (n053_score.py -> n053_baseline.csv): n051_v10_base .6730 (b10 .6867 / b01 .6592); n048_v9_base
  .6742; n051 base+zoom2 ensemble .6788; GOAL best .6804 (N048 ensemble).
- Weights: DINOv3 timm/vit_{small,base}_patch16_dinov3.lvd1689m (ungated timm mirror, DINOv3 licence) -> models/hf; DinoBloom-B
  models/DinoBloom-B.pth; VSCyto2D (BSD-3) converted to timm convnextv2_tiny by an agent (stem 2x2/s2, no dw bias, eps 1e-5; numerically
  matched to a re-implementation, max rel err 1.3e-6) -> models/cytoland/converted/. DynaCLR 2D only as ONNX, conversion unverified -> not used.
- Smoke 1376787/8 (30k cells, 1 epoch, b1to0): dinov3_b 61 s, dinobloom_b 49 s per epoch-pass incl. eval; r_mean_valid .61 after 1 epoch.
- Codex red-team of the encoder code (results/N053-*/codex_rt/report.md; probe.py, evidence.txt): no P0/P1; pretrained weights load
  (DINOv3 S/B 162/162 tensors, DinoBloom 174/174, Cytoland all source tensors), train/eval rows identical to the r50 baselines, eval
  unchanged, OneCycle keeps the 0.1 ratio. Dispositions:
  F1 normalisation stats (led_norm_stats.py) sample every slide incl. held-out-draw cells: REPRODUCED first-hand (per-slide
     background-ring flat-field scale from 300 random cells/slide; global s_c from 2,000 cells). Label-free per-slide instrument
     calibration, shared by every GOAL model since N005 -> disclose as transductive calibration; no change now (TODO: source-only control).
  F2 resume lacks RNG state / label-content hash (inherited): [unverified], no change (affects only preempted runs).
  F3 encoder lr x0.1 also on the new gate/patch-embedding, pooling avg (DINOv3) vs CLS (DinoBloom), resolution differs: valid caveat ->
     a negative result rates THIS transfer recipe, not ViTs; one lr/pooling follow-up on the best arm before concluding.
  F4 result.json r_mean_markers_valid != GOAL (inherited): known; n053_score.py scores common-11.
  F5 Cytoland eps: downsample norms are 1e-6 in VisCy, my blanket 1e-5 gives max feature error 1.4e-4 at init (REPRODUCED in
     evidence.txt lines 32-34). Running jobs keep it (negligible after fine-tuning); fix bsccm/cytoland.py after they finish.
  F7 weight decay on gate/norm/tokens: minor (decay factor ~.97 over the run at enc lr), no change.
- Full runs 1376828-35 (jobs.txt): 4 archs x {b10, b01}, 10 epochs, seed 0; 6.2 GPU-h actual (27-71 min per run).
- RESULTS v10 GOAL single model (n053_scores.csv): r50 base .6730 | Cytoland .6671 | DinoBloom-B .6552 | DINOv3-B .6515 |
  DINOv3-S .6495 -> no encoder beats the r50 alone (pre-registered primary criterion FAILED).
- Ensembles (n053_scores.csv, n053_controls.csv): N048 (.6804) + Cytoland = .6862 (b10 .7001 / b01 .6723, both up); + DinoBloom
  .6816, + DINOv3-B .6811, + DINOv3-S .6803. Same-size controls (N048 + one r50-family / ConvNeXt member): .6773-.6826 (best
  n040_student_base). Residual r vs N048 base: Cytoland .911, ConvNeXt .886, ViTs .92-.93, r50 family .95-.99 (n053_decorr.csv).
  Paired FOV bootstrap (500): E3(+Cytoland) - E3(+best r50 member) = +.0036, 95 % CI [.0007, .0062]; DinoBloom -.0010 [-.0047, .0026].
  Caveat: Cytoland was chosen post hoc among 4 third members -> confirmation runs below before any GOAL claim.
- Learned LED gate x patch-embed norm (mean over directions, relative to mean): ViTs flat (+-10-20 %), consistently highest for
  high-angle darkfield DF_90 / DF_85 / DF_80 and LED119; Cytoland DF_90 2.0x, DF_85 1.37x, phase 1.28x (cf. N018: darkfield carries
  CD123). Descriptive only (gate is a reparameterisation under weight decay).
- Follow-ups 1377120-23 (n053_followup.csv): Cytoland seed 1 (eps fix F5 applied) single .6649 (seed 0 .6671); N048 + Cytoland s1
  = .6864 (b10 .7043 / b01 .6685) -> the ensemble gain REPLICATES (s0 .6862). Cytoland enc-lr-mult 1.0 (r50 lr, red-team F3):
  single .6464, ensemble .6810 -> the low encoder lr is right. N048 + both Cytoland seeds (4 members) .6880 (post hoc, not the GOAL value).
- Plots: .agent-research/plots/goal-v10-ensemble_2026-11-18_0912.png, goal-v10-single-encoder_2026-11-18_0912.png
  (plot_n053.csv, plot_n053_single.csv).

## Outcome
IMPROVED: GOAL v10 .6804 -> .6862 (N048 r50 ensemble + one Cytoland VSCyto2D member; seed-1 replicate .6864; > 2 x noise).
Honest decomposition: ensemble size alone (best same-size r50 member) gives .6826; the Cytoland-specific part is +.0036
(FOV-bootstrap CI [.0007, .0062]) -- the microscopy-pretrained ConvNeXtV2 makes the least correlated errors (resid r .91) while
staying accurate. No encoder beats the r50 as a single model (Cytoland -.006, ViTs -.018..-.024): the pre-registered primary
criterion failed; the gain is ensemble diversity. Natural-image (DINOv3) and blood-smear (DinoBloom) ViTs add nothing.

## Outputs
<!-- files in .agent-research/results/N053-slug/ + external absolute paths -->

## Next
