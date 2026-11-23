# N074 · Dense privileged supervision: auxiliary decoder predicts each training cell's 6-channel fluorescence crop (local background, antibody-panel conditioned) from the LF encoder; scalar v10 heads unchanged

type: exp · date: 2026-11-19 · status: done
goal_effect: flat · scale: paradigm · value: 0.6683

## Goal
Review N066 paradigm bet (lit_labelfree idea 2, Codex N066 idea 4, N059 parked "privileged supervision"): change what the loss sees.
Every attempt so far supervises 11-13 scalars per cell; the fluorescence crop of each training cell (~10^4 values, local background,
no LOWESS field error) has never been a training signal. Literature: dense auxiliary targets help the ENCODER (in-silico labelling,
supervised autoencoders), while reading markers from generated images is worse (HEmIF-Bench -.093 AUROC; in-project K027/N034).

## Plan
Codex (write hat, results/N074/codex/, brief.md): pack_fluor_aux.py -> /hnvme/workspace/v181bb10-bsccm/fluor_aux64.npy (float16, all
rows, 6x64x64, local corner background, per-draw asinh + standardisation on split-0 cells); train_lf.py --dense-aux PATH --dense-w w:
FiLM(antibody condition)-conditioned upsampling decoder on the r50 pre-pool feature map, masked per-pixel MSE on training-draw cells
only; flag off = legacy numerics. Claude red-teams the landed code (mutual review) before any GPU run.
Arms: dense (w .5) on the GOAL r50 recipe (n051_v10_base: SSL ex0/ex1, 10 ep, lr 5e-4, bs 256, w-cls .5), seeds 0,1, both directions;
~4 x .6 GPU-h + CPU packing. PRIMARY (pre-registered): mean GOAL (v10, seeds 0+1) vs control .6706; success if delta > +.0054.
Expected +.002..+.008 (lit). Kill: <= .0027, or member error correlation with N053 members >= .94 with no GOAL gain.

## Changes
Codex (write, codex/README.md; 7 CPU tests incl. flag-off byte equality vs the pre-N074 harness) -> Claude red-team
(agents_redteam.md): no BLOCKER; MAJOR-1 fluorescence crops are offset from LF by a per-channel instrument shift (red-team cross-corr
(0,+2)..(-1..-2,+6..+7) px; reproduced first-hand by centre of mass: fluor - phase dx +.3..+2.8, dy -.5..-1.1) -> the anti-diagonal
reflection (N068) would give contradictory targets; MINOR: Giemsa-paired cells in packer statistics (reproduced 61 %/12 % of split-0
unstained), grad clipping shared with the decoder, main-loop I/O, packer import path, chunk size.
Integrated (Claude): scripts/train_lf.py, bsccm/dense_aux.py, scripts/pack_fluor_aux.py with fixes: --dense-aux refuses --diag-aug
(MAJOR-1; per-channel shift correction = future work), decoder clipped separately + dense_loss logged per epoch, packer excludes
hist_row>=0 from cofactor/standardisation rows, imports from the repo root, chunk-size 512. Re-ran Codex's tests on the integrated
files against the reconstructed pre-N074 harness (sha 88f70936): 7/7 OK (verify/integrated_tests.log; two tests had --diag-aug removed
because the combination is now refused). I/O throughput of the main-loop target read is unmeasured (GPU).

## Evidence
Packer v1 1391847 failed (self-hash path after moving to scripts/; fixed, smoke-tested on the Codex fixture); v2 1429986 COMPLETED
(1:48 min; fluor_aux64.npy 20.3 GB float16 + json; per-draw cofactors 1.5-6.7 ADU). Dense runs 1429987-990 exit 0 (b10 ~55-60 min vs
control ~31 min: the main-loop target read costs ~2x on the large draw; b01 16 min).
Auxiliary loss falls (b10 .356 -> .207, b01 .447 -> .241 per standardised pixel) -> the decoder learns the fluorescence images.
RESULT (scripts/score_goal_arms.py, v10): dense GOAL .6683 (b10 .6850, b01 .6516) vs control .6706 -> -.0023 = FLAT
(b10 +.0040, b01 -.0087). Per target within +-.02 (CD123 .539 -> .522, CD3 .768 -> .775, grp:CD45 .690 -> .695); MM3 ~flat.
Dense pixel supervision of the same cells' fluorescence does not improve the scalar marker readout (consistent with lit: in-silico
labelling helps new-label learning, not readout accuracy on labels the head already sees).
Plot: plots/goal-r-n074_2026-11-23_1600.png

## Outcome
<!-- on close: bin/effect.sh N074 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N074-slug/ + external absolute paths -->

## Next
Dead on BSCCM as an auxiliary loss. Keep the code (flag off = legacy). The 9-image IFC variant (Codex N079 idea 2) only if N080/N081 show IFC transfer.
