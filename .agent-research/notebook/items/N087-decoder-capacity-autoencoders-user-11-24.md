# N087 · Decoder capacity + autoencoders (user 11-24): rank-k / MLP / kNN decoder sweep on frozen features; marker-autoencoder co-expression prior in the S-BIAD452 sandbox; low-dimensional image autoencoder for clustering and GOAL vs SSL features

type: exp · date: 2026-11-24 · status: done
goal_effect: flat · scale: module · value: 0.6700

## Goal
User 11-24: "have you tried a lower dimensionality autoencoder, and how did you decode -- is this also something we might need to
optimize?" -> "OK try both of that [rank sweep + marker-AE prior], also a low dimensionality autoencoder on the images might give
better clustering??" (marker-AE prior = learned co-expression prior, explicitly OK'd; memory feedback_bsccm_marker_ae_prior_ok).

## Plan
A. Decoder capacity on FROZEN features (n051_v10_base seed 0 fine-tuned ResNet-50, both directions; plus frozen SSL encoder and AE latents):
   model's own head (must reproduce .6730) vs ridge vs reduced-rank k = 1..128 vs MLP 2x512 vs kNN. Hypothesis (from N060-N062 flat
   heads + N082 label-limited): fine-tuned features -> rank <= 13 recovers the linear head and MLP adds <= +.003 (decoder is not the
   lever). If MLP/kNN > linear by > .0054 on both directions -> decoder capacity matters -> end-to-end nonlinear head run.
B. Marker-AE prior in the S-BIAD452 sandbox (ground-truth joint labels): single-marker regime with predictions forced through a frozen
   linear (PCA) or AE decoder (k = 2/4/6) fitted on the validation donor's full labels. Compare with N080 single / full. Success:
   registered 36-pair error and within-type r move toward 'full' (pair error <= .21 vs single .229) without macro r loss > .01.
   Only then port to BSCCM (prior from S-BIAD452 overlap markers + BSCCM pooled cells).
C. Image autoencoder (k = 16/64/256) trained on the training draw only: clustering of held-out draw-0 cells vs gated cell classes
   (k-means NMI/ARI, kNN balanced accuracy) and GOAL from frozen latents, vs SSL ResNet-50 features (2048-d and PCA-k).
   Expectation (literature: pixel AEs encode intensity/background, SSL encodes semantics): AE clusters worse than SSL.
D. User 11-24 follow-up ("look at the pretrained autoencoders ... make them smaller from the start or with a projection layer"):
   structures: SSL ResNet-50/ConvNeXt-S = encoder -> pooled 2048/768-d -> projector MLP 2048-2048-2048-256 (BN, GELU; SSL loss on the 256-d
   output; dropped at fine-tuning, weights kept in the ssl run's ckpt.pt['projector']); Cytoland VSCyto2D = UNeXt2 encoder-decoder (FCMAE masked-
   autoencoder-pretrained ConvNeXtV2-T encoder + 72-tensor decoder, we use only the encoder); DynaCLR ConvNeXt-T (converted, never tested).
   D1 frozen projector outputs (256-d and intermediate 2048-d layers) in the decoder sweep; D2 end-to-end fine-tune with a projection
   bottleneck (train_lf.py --bottleneck 32 fresh Linear+LayerNorm, --bottleneck-init projector = pretrained SSL projector, 256-d), seed 0,
   both directions, ~2 H100-h; D3 narrow SSL representation from the start only if D1/D2 show low-dim features are not worse (~6 H100-h
   per direction). Claude added --bottleneck/--bottleneck-init (default off, _NEW_DEFAULTS) -> Codex red-team before any run.
E. User 11-24 ("align the distribution of image latents with marker latents ... flow matching?"): flow matching so far = N034 (LF ->
   fluorescence images; per-cell r .476 vs regression .576, CRPS .674 vs .815), N044 (residual flow: r .594 vs .599, CRPS .591 vs .778,
   under-dispersed on the new draw sd ratio .72; failed its rule), N054 (Giemsa <-> LF+marker latent flows .642 vs direct .647; oracle
   flow .794/.640 = flow works, Giemsa encoder is the bottleneck; unpaired GW flow .32), N034-B set matching. NOT done: generative
   decoding of the JOINT marker vector from the image latent. Rationale: the MSE head predicts the conditional mean, which erases
   co-expression when the image is ambiguous (CD4 vs CD8 in T cells -> 'half of each'); a conditional flow samples a whole vector.
   E1 (sandbox, joint labels exist): conditional flow-matching decoder p(9 markers | frozen image features) vs MSE decoder on the same
   features, full regime; metrics per-cell r (sample mean and single sample), registered 36-pair error, CD4-CD8 in T, gated population
   fractions vs truth. E2 only if E1 restores co-expression: single regime with the flow in the marker-prior latent (prior from the
   validation donor's joint labels), then BSCCM via S-BIAD452 / pooled-cell priors. < 1 H100-h.
F. User 11-24 ("have you tried to align Giemsa with markers?" -> "OK can we also try this"): no same-cell Giemsa-marker pairs exist (N004);
   only unpaired N054 A2 (.32, failed) and the LF bridge (distillation .66; N017 CIA estimate CD16 .80 ... CD123 .17). CLASS-CONDITIONAL
   matching: F1 Giemsa cell-type classifier (N008-style PBC/DinoBloom or anchored student classes) -> class-mean marker profile (classes of
   marker cells from cls10 gates on pooled cells and the LF model's c10 head on single-stain cells); F2 within-class OT between Giemsa
   embeddings and marker profiles of the same class. Truth on Giemsa cells exists only for autofluorescence -> report per-marker agreement
   with the LF teacher on the same 15.7k cells, autofl r, MLL23/PBC contrasts; baseline = distillation (TT6c/hub). Sandbox analogue:
   S-BIAD452 type-oracle .814 vs image models .596. Mostly CPU on existing features.
Compute: dumps 2 x ~10 min; AE 3 dims x 2 directions ~6 H100-h; sandbox 3 priors x 3 dims x 3 donors ~2.5 H100-h; total < 10 H100-h
(preempt). Writer: Codex (brief .agent-research/results/N087-decoder-capacity-autoencoders-user-11-24/codex/brief.md); Claude red-teams + integrates; --dump-features added to train_lf.py by Claude.
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes

## Evidence
- Codex writer (codex/, 49 CPU tests) -> Claude red-team (redteam_claude.md): NOTHING BREAKS, 7 MINOR. Integrated fixes in
  codex/n087_decoders.py (orig kept as n087_decoders.codex_orig.py): per-target loss weighting as train_lf (m1: k=2 .407 vs .507),
  wider ridge grid + logging (m2), reference assert only for n051 dumps (m3), ranks 1,2,4,8,12,full (m4); clustering on all 10,137
  draw-0 cls10 cells via runs/n087_cluster_rows_b10.npz (m5); plain AE --mask-fraction 0 (m6, user asked for a plain AE); m7 noted.
- Jobs (n087_job.sbatch, 1 x H100 preempt each): 1442606 decoders (SSL extraction + sweep on fine-tuned and SSL features);
  1442607-12 AE k=16/64/256 x b10/b01; 1442613 sandbox marker prior linear/ae k=2/4/6 x 3 donors + eval; 1442504 flow decoder
  (Claude script n087_flow_decoder.py); 1442617-19 N088-B control: sandbox FULL regime on 1/9 of the training cells x 9 epochs
  (equal label count and steps as 'single') -> is N080's full-vs-single gap joint measurement or label count?
- E1 flow decoder (job 1442504, results flow_decoder.csv): sandbox mean of 3 donors, frozen full-regime features: per-marker r linear head
  .588 / MLP .549 / flow mean .554 / flow one sample .484; CD4-CD8 in T (true -.804) -.36 / -.27 / -.22 / -.31; 36-pair error .189 /
  .215 / .224 / .197; sd ratio .74 / .82 / .82 / .97. BSCCM GOAL on frozen n051 features: model .6730, flow mean .6507, MLP .6369
  (flow vs MLP confounded: 2.1M vs 1.6M params, 2x updates on BSCCM; Codex red-team MAJOR). Reading: generative decoding fixes the
  spread but not co-expression; the limit is the features. Not a GOAL route.
- N088-B equal-label-count control (jobs 1442617-19, fullninth_eval/): FULL labels on 1/9 of the training cells x 9 epochs (same label
  count and steps as 'single'): macro r .577 (full .596, single .526; per donor .471/.694/.565 vs single .336/.622/.619), registered
  36-pair error .174 (full .189, single .229; better than single on all 3 donors), CD4-CD8 in T -.40 (full -.36, single -.07).
  Reading: most of the full-vs-single gap (+.051 of +.070 macro r, and all of the co-expression gain) comes from measuring markers
  JOINTLY, not from more labels; macro r is not consistent across donors (donor 2 reversed), co-expression is.
- A decoder sweep (job 1442606, decoders/goal.csv): fine-tuned n051 features: model head .6730 (reference reproduced), ridge .6700, kNN
  .6637, MLP .6605, reduced-rank k=12 .6635 / 8 .6562 / 4 .6166 / 2 .5398 / 1 .4580 (Adam 'full' linear .6127 = optimisation artefact;
  ridge is the full-rank linear). Frozen SSL features: ridge .5950, MLP .6027, kNN .5660, k=8 .5835. Reading: no decoder beats the
  end-to-end linear head; predictions need ~8-12 dims; fine-tuning adds ~+.07 over frozen SSL features, a nonlinear head on frozen
  features recovers +.008 of it. Decoder capacity is not the lever (as predicted).
- B marker prior (job 1442613, sandbox_eval/; mean of 3 donors; single baseline macro .526 / 36-pair .229 / CD4-CD8 -.07 / within-type
  .176; full .596 / .189 / -.36 / .196): linear2 .501/.346/-.33; linear4 .570/.270/-.39/.139; linear6 .556/.259/-.12; ae2 .507/.229/-.24;
  ae4 .483/.240/+.07; ae6 .533/.212/-.27/.161. Pre-registered success (pair error <= .21, macro loss <= .01): NONE (ae6 closest .212).
  Linear4 restores CD4-CD8 on average and lifts macro r (+.044, mostly donor 0) but worsens the 36-pair error; within-type r drops for
  every prior. Donors disagree strongly (n = 3). Reading: a prior learned from another donor's joint labels does not transfer cleanly;
  no BSCCM port.
- C image AE (jobs 1442607-12 k=16/64/256 x b10/b01, plain AE mask 0; post 1444242 decoders ok, clustering rerun on CPU without the
  fine-tuned dump rows whose cohort differs): clustering of 10,137 held-out draw-0 cells vs cls10 gates (cluster/): SSL r50 NMI .340 /
  ARI .205 / kNN bal-acc .306 (PCA 16-256 identical); AE16 .394/.250/.342; AE64 .390/.260/.346; AE256 .381/.254/.344. CONTROL
  (n087_stats_baseline.py, 144 hand-made per-channel image statistics, no learning): .351/.269/.354 = the AE's advantage is mostly
  brightness/scatter/contrast; SSL features cluster worst (SIGReg isotropy). Frozen-feature GOAL (decoders_ae/): AE64 MLP .6235, AE16 MLP
  .6173, AE256 MLP .5944; AE ridge .515-.547; vs SSL MLP .6027 / ridge .5950; fine-tuned .6730. Statistics baseline for GOAL (CPU jobs
  1444332/33, decoders_stats/): MLP .6071, ridge .5634, kNN .5508 -> frozen SSL features carry no more marker information than
  hand-made image statistics; AE64 +.016 above them; fine-tuning +.05-.07 over every frozen representation.
- Codex red-team of Claude's code (train_lf --dump-features/--bottleneck, n087_flow_decoder.py, decoder integration): codex_rt/.
- Jobs: 1442342/1442343 n087_feat_{b10,b01} (--dump-features on n051_v10_base seed 0; 1 x H100 preempt, inference only).
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
FLAT (decoding): no decoder on frozen features beats the fine-tuned linear head (ridge .6700 vs .6730); predictions need ~8-12 dims.
Plot: .agent-research/plots/goal-direction-r-frozen-decoders_2026-11-27_0725.png
Autoencoders cluster cell types better than SSL features but only as well as raw image statistics; marker priors and the flow decoder
miss their bars; joint measurement (not label count) carries co-expression. Not run: D1 projector features / D2 bottleneck fine-tune /
D3 narrow SSL (superseded by N088 lit A: post-projector features are worse; the decoder sweep shows low-rank costs accuracy), E2, F
(class-conditional Giemsa->markers; moved to IDEAS). Figure 10: figures/overview/fig10_decoders_autoencoders.png.
<!-- on close: bin/effect.sh N087 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N087-slug/ + external absolute paths -->

## Next
