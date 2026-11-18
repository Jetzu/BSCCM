# N054 · Tri-modal latent: user two-space design (LF+marker-matrix SigLIP space, LF+Giemsa space, flow-matching alignment) vs LF-anchored hub (latent distillation) vs output-distilled student; Giemsa metrics + hidden-information marker completion

type: exp · date: 2026-11-18 · status: done
goal_effect: flat · scale: paradigm · value: void

## Goal
User ruling N052: test the user's two-space design (LF+marker-matrix space, LF+Giemsa space, flow-matching alignment, markers
decoded from the matched space) against the LF-anchored hub (latent distillation) and the current output-distilled TT6c student;
test the user's hidden-information idea (marker completion; does conditioning on observed markers add information beyond LF?).

## Plan
Writer: Codex (brief codex/brief.md), Claude red-teams + integrates. Fixed: frozen r18 teacher LF features (v4c markers + v6c
autofl, as TT6c), targets_tt6c labels, DinoBloom-B GS encoder (freeze 8, 2500 steps), cross-slide holdout batch1 and batch0.
Arms: A1 two-space + paired LF-bridge flow (z2 -> z1); A2 two-space + unpaired minibatch-OT flow (marginals only);
A3 hub (GS encoder regresses the LF latent, SigLIP weight 0 / .5); A0 TT6c re-evaluated with the same code.
Completion: C1 autofl from LF vs LF + observed marker; C2 pooled leave-one-group-out; C3 coherence table.
Metrics: M1 same-cell autofl truth, M2 student-vs-teacher x teacher-truth, M3 retrieval, M4 CKA / mutual kNN, M5 LF->MK through
each space, M6 PBC + MLL23 pre-registered contrasts.
Predictions (Claude, before results): A3 >= A0 > A1 > A2 on M1/M2/M6; A1/A2 retrieval and CKA fine but marker info low (N045);
C1/C2 fused > LF-only on held-out FOVs, smaller gain across draws.
Not a GOAL attempt (GOAL = LF->MK); effect void unless an LF-side result changes the GOAL.

## Changes

## Evidence
- Codex v1 delivery (codex/: n054_trimodal.py etc.; CPU smoke passed, A0 reproduces TT6c preds r >= .9993). Codex flagged: tt6c labels
  have autofl only on unstained cells -> C1 impossible there; targets_v5cx_v10.npz has autofl on all 312,590 single-marker cells
  (verified: single&autofl 312,590) -> C1 moves to v10 labels.
- Claude red-team of v1 (general-purpose agent; scripts scratchpad/n054rt/c1-c13.py): no P0 leakage (splits audited by running).
  P1s REPRODUCED first-hand: (1) modality gap: TT6c test cells cos(z_G, own z_L) .318 +- .233, ||z_G - z_L|| 1.15 vs NN .37 ->
  flow trained on P2(f) but run on z_G; (3) A3 MSE mean over 256 dims: .5 SigLIP gradient 453-562x larger (c11); (5) C2 labels-only
  baseline (other pooled groups) r .840/.776/.747 vs LF-only .853/.838/.847 -> fused > LF-only guaranteed without a labels-only arm
  (c10); (2) A2 target E_M(full rows) never seen by D_M (read lines 370-382, 464-472); (6) sbatch uses scontrol requeue, not
  preempt-run (grep). (4) identical models retrained per config (submit graph) -> unpaired noise.
- v2 fix brief (codex/v2/brief_v2.md): A1-LF + A1-GS flows + oracle/direct-MLP diagnostics; D_M on full rows; A2 = unpaired OT z_G ->
  P1(f) of other cells (+ A2-EM secondary); A3 sum-MSE; shared models trained once; labels-only + ridge baselines for C1/C2 (C1 per
  marker for dye crosstalk); 3 preempt-run driver units; scripted aggregation from the frozen selection; versioned runs/n054v2_*.
- Codex v2 delivered (false CODEX-UNAVAILABLE gate; handback + smoke OK, 156 s CPU, SIGTERM resume bitwise). Integrator: removed
  `#SBATCH --mem` (cluster rejects --mem for GPU jobs). Claude re-review of v2 launched before the GS units.
- LF unit 1377051 (3 min, H100): completion C1-C3 on v10 labels (results/N054-*/completion.csv, coherence.csv), seed 0, 1 run:
  C1 autofl, held-out FOVs: fused .689 vs LF .526 / labels-MLP .524 / ridge[f+labels] .556. Per observed marker the big gains sit
  where the observed value alone predicts autofl (CD56 .200 -> .659, labels-MLP .537; CD3 .249 -> .655, .525; CD19 .417 -> .608,
  .580) = the dim dye channel shared by CD3/CD19/CD56 (K028 background) -> crosstalk/shared measurement error, not biology. Markers
  without labels-only signal (CD14, CD16, HLA-DR, CD123, CD45): +.02-.06 over the best control in-draw; across draws mixed
  (b0->1 +.005..+.03, CD45 -.21; b1->0 CD45 -.17, HLA-DR -.05).
  C2 pooled leave-one-group-out: in-draw fused beats the best control by +.001..+.022 (CD45 .022, CD16 .017); across draws
  <= 0 except CD16 b0->1 +.018. The gain over LF-only (e.g. CD3/19/56 .834 -> .882) is label correlation a ridge on [f, other
  groups] already captures (.874).
  C3 coherence (top-5 % observed, FOV): CD14-high decoded HLA-DR +2.36 SD, CD45 +1.25, CD16 -.88 (monocyte-coherent; LF-only
  +.90/+.55/-1.04); CD19-high decoded CD45 +.76 (LF-only +.39) but CD3 +.57 and HLA-DR only +.12 (B cells are CD3-, HLA-DR+:
  incoherent = shared-dye artefact); CD56-high decoded CD3 +1.25, CD16 -1.06 (shared dye again).
  Reading: conditional completion through the shared space works mechanically, but beyond (i) label correlations a linear
  model captures and (ii) shared-dye measurement error, the extra information is small within a draw and absent across draws.
- Claude re-review of v2 (scratchpad/n054rt2/d1-d7.py): no P0; all v2 fixes implemented; preempt SIGTERM -> exit 143 reproduced by
  the reviewer. P1s REPRODUCED first-hand by rerunning d2/d6/d7:
  A in-sample maps: legacy same-recipe E_G cos(z_G, z_L) .717 on its training cells vs .558 on held-out FOVs (R@1 .54 vs .20/.23)
    -> A1-GS/direct trained on split-0 z_G never see the test-time gap.
  B A2 Euclidean OT across independently initialised spaces: identity 0.000 and matched-cell autofl r -.03 after a frame rotation
    (hub vs space1 only matched because of shared seed/init, cos .878) -> A2 null by construction.
  C completion: a linear stack of LF-only + labels-MLP (fit on half the test set, optimistic for the stack) gives fused - stack
    -.078..+.050 on C1 per marker (CD19 -.078, CD14 -.009, CD45 -.003; CD56 +.050, CD3 +.041) and +.017..+.032 on C2 (FOV).
    The C1/C2 numbers above therefore OVERSTATE the shared-space gain; v3 adds stack + plain-MLP controls fitted on source split 1.
  P2: test-draw non-GS unstained LF cells in LF-side training (~2 %; disclosed); __pycache__ under -I (use -B); signature change
    forces an LF rerun (prefix n054v3).
- v3 fix brief (codex/v3/brief_v3.md): out-of-sample (split-1) maps with LF-pair pretraining for both A1-GS and direct; A2 primary
  = minibatch entropic Gromov-Wasserstein coupling (+ identity sanity row); completion controls (e) stack and (f) plain MLP.

## Outcome
RESULTS (v3, seed 0, units 1379287 LF / 1379327 batch1 / 1379328 batch0, ~3.2 GPU-h; aggregation from the frozen selection ->
results/N054-*/tables/, n054_summary.py -> n054_arm_summary.csv; plots plots/n054-gs-autofl-truth-r_2026-11-18_1237.png,
n054-gs-marker-r-est_2026-11-18_1237.png). Held-out Giemsa slide, mean of the two holdouts (batch1 / batch0):
| arm | M1 same-cell autofl r | M2 marker r_est (CIA, 12 markers) |
| A0 TT6c output-distilled student | .658 (.715 / .601) | .497 |
| A3 LF hub, latent distillation | .650 (.751 / .549) | .516 |
| direct map z_G -> space 1 (no flow) | .647 (.747 / .546) | .506 |
| A1-LF two-space + LF-bridge flow (user design) | .642 (.702 / .581) | .508 |
| A1-GS two-space + GS-pair flow (out-of-sample) | .625 (.700 / .550) | .512 |
| A2 two-space + unpaired entropic GW flow | .322 (.180 / .463) | .406 |
Diagnostics: oracle flow from P2(f_own) (perfect Giemsa embedding) .794 / .640 -> the flow itself works; the Giemsa->LF encoder gap
(cos(z_G, P2 own) .51 / .60) costs ~.09. GW identity sanity on the SAME LF cells encoded in both spaces: 1.3 % / 2.0 % (chance .39 %)
-> unpaired GW cannot align the spaces at cell level (identifiability, cf. N034-B). Fine-tuning maps on out-of-sample GS pairs
changes little (pre/post .701 -> .700, .555 -> .550). Clinical (M6) uninterpretable for all arms: these cross-slide students have
no PBC/MLL23 anchoring, lineage contrasts at or below chance (inverted CD3/CD45) incl. TT6c (K016: anchoring needed) -> E3.
Completion v3 (all controls incl. stack + plain MLP fit on source split 1): within draw the shared space beats every control by
small margins (C1 aggregate +.044 over the stack .645; per marker -.08..+.07, CD19 negative; C2 +.000..+.023); across draws
mostly <= 0 (C1 per marker 3/8 positive, CD45 -.18/-.21; C2 -.09..+.01) -> the extra "hidden information" is draw-specific.
CONCLUSION: with pairs through LF, the user's two-space design (A1) performs like the hub and the TT6c student (all within ~.03
on autofl, single seed; latent arms +.01-.02 on the CIA marker estimate); the decisive ingredient is pairing, not the number of
spaces; truly unpaired alignment (A2) fails. Not a GOAL attempt (GOAL = LF->MK) -> effect void.

## Outputs
<!-- files in .agent-research/results/N054-slug/ + external absolute paths -->

## Next
