# Outsider view: LF->MK v10 at .6862, target .70 (gap +.0138 = 5.1 noise sd)
Sources read: GOAL.md, attempts.md, declined.md only. GPU-h figures assume one standard SSL-r50 member costs about 2-3 H100-h
per direction. Scale them by your logged step time.

## Read of the situation
- Nine attempts all changed the predictor. Ensemble gains have dropped to +.002 per member, and errors correlate r .94-.96
  even across architectures (N050). When models trained on the same labels share 95 % of their error, the remaining error is
  probably on the label/biology side (or comes from the shared loss), not variance between models.
- The cross-draw ensemble (.6862) equals the in-distribution single model. So .70 needs either an in-distribution
  ensemble >= .70 with almost no domain gap, or a new source of information or supervision. More members of the same kind
  will not get there.
- My prior: .70 is at or above the practical ceiling of the current recipe. I put about 30 % on reaching it with the ideas
  below. Measure before spending (idea 0).

## 0. Measure the ceiling first (decides whether ideas 1-4 are worth running)
| measurement | how | cost | decision rule |
|---|---|---|---|
| M1 same-recipe limit | fit r_N = r_inf - c/N to ensembles of 1..N existing members, per direction and per marker | CPU, minutes | r_inf < .692 means more members or architectures are dead; only a new loss or new supervision can help |
| M2 measurement-noise ceiling | split each raw fluorescence crop into two checkerboard pixel halves, run the v10 label pipeline on each half, take the split-half correlation, apply Spearman-Brown, then r_max,k = sqrt(rel_k) | CPU, a few hours | an upper bound (shot and read noise only). If the mean of r_max over the 11 targets is < ~.75, .70 needs >93 % of ceiling: call it unreachable and report r / r_max |
| M3 learning curve | r50, 1 seed, n in {1/8, 1/4, 1/2, 1} of the source draw; score in-distribution and cross-draw; b0to1 first | ~4 runs, about 6-10 GPU-h | per-marker slope per doubling at full n. If the mean slope is < .003 and the fitted asymptote is < .70, the target is unreachable with this data. If dim-marker slopes are >= .01, they are short of labels: run ideas 2-3 |
| M4 rare-event leverage | per marker: number of positive training cells per draw; share of Var(y_test) and of Cov(pred, y) from the top 1 % / 5 % of labels; r on winsorised labels | CPU, minutes | if the top 2 % carries > 50 % of Var(y) for CD19/CD56/CD123/HLA-DR, the dim-marker r is a few-shot rare-positive detection problem (maybe only 100-800 positives per draw). That points to ideas 2-3 |

## 1. Distributional (HL-Gauss histogram) head in place of MSE
- **Mechanism:** predict a 64-128-bin categorical distribution over log-intensity, trained with a Gaussian-smoothed
  cross-entropy (HL-Gauss, Farebrother et al. 2024; same idea as DEX/DORN in age and depth regression). The read-out is
  E[v10 label | x] = sum over bins of p_b * asinh(I_b / c_v10). That expectation is the Pearson-optimal prediction in the
  metric's own space. The target choice is also learned, so N048-style label-transform tuning becomes automatic.
- **Why it could help here:** dim-marker targets are bimodal mixtures with noisy bright tails. MSE gradients are dominated by
  a few tail cells. Cross-entropy over bins is bounded, models the negative/positive mixture natively and is known to scale
  better with noisy targets. All current members share the loss, which plausibly explains the .95 error correlation. A
  different loss should decorrelate errors more than a different backbone did.
- **Expected gain:** +.003-.008 for a single model, +.002-.006 in the ensemble, mostly on the dim markers.
- **Falsifier:** an MSE twin vs an HL-Gauss twin of the SSL r50, same seed, both directions. 2 runs, about 5 GPU-h. Drop it
  if the single-model delta is < +.003. For a GOAL claim, require at least +.0054 in the ensemble.
- **Build:** about 0.5 day (a new head and loss; the read-out is a dot product).

## 2. Assay measurement-model pooling: a latent 8-marker head with the known panel matrix
- **Mechanism:** every cell has one assay. Single-antibody tubes and the pooled-panel groups are different noisy views of the
  same 8 latent markers. Predict z_m(x) for all 8 markers on every cell and model each assay as
  y_a = g_a(sum_m W_am z_m) + noise. W comes from the panel design and spillover, which is known physics, not a lineage
  prior. g_a is a learned monotone per-assay link. Panel-tube cells then supervise the single dim markers: for example,
  the draw-0 grp:CD123/HLA-DR/CD14 cells, which are out of the metric but valid labels, can supervise the CD123 and HLA-DR
  latents. Single-tube cells in turn constrain the group targets.
- **Expected gain:** about s_k * log2(1 + n_panel,k / n_single,k), where s_k is the marker's slope per doubling from M3. For
  rare-positive markers where M3 or M4 shows label starvation, I estimate +.01-.03 per affected marker, which is
  +.004-.01 on the mean. If M3 slopes are flat, the gain is about 0.
- **Falsifier:** M3 and M4 first (CPU plus the M3 runs). Then 1 run per direction, about 5 GPU-h.
- **Build:** about 1 day (the W matrix from the panel sheet, the link functions and the masked loss).

## 3. Rare-positive hurdle head for the dim markers (do this if M4 says "few-shot")
- **Mechanism:** E[y|x] = p(pos|x) * mu_pos(x) + (1 - p) * mu_neg(x). The gate is trained with class-balanced or focal
  cross-entropy, then corrected back to the source-draw prior so it stays Pearson-optimal. MSE is optimal only in theory.
  With a few hundred positives it under-fits the rare morphotypes (basophils and pDC for CD123, NK for CD56, B cells for
  CD19), and those cells carry most of the test variance.
- **Expected gain:** +.01-.02 on 2-4 markers, which is +.003-.008 on the mean. It stacks with idea 1: the histogram head is
  the soft version of this.
- **Falsifier:** an oracle headroom check on existing predictions, CPU only. Set the predictions of the true top-q cells to
  their source-draw positive mean and recompute r. If the headroom is < .02 per marker, kill the idea. Otherwise run 1 run
  per direction, about 5 GPU-h.
- **Build:** about 0.5 day.

## 4. Per-target weights and a cross-target stacker fitted on source-draw out-of-fold (OOF) predictions (cheapest)
- **Mechanism:** today's ensemble is presumably a uniform average. Instead, fit per-target member weights plus a small
  GBM/ridge model on all 12 heads' outputs and the members' disagreement, using source-draw OOF predictions only, then
  apply it frozen to the other draw. It exploits members that are better on particular markers (Cytoland vs r50).
- **Caveat:** cross-target inputs are a learned joint structure. Ask the user whether this counts as a "lineage prior"
  (declined 11-17). If it does, keep only the per-target weights and the isotonic step.
- **Expected gain:** +.002-.005.
- **Falsifier:** 0 GPU-h if in-distribution held-out predictions exist. Otherwise 2-fold cross-fitting costs about 2 members
  per direction.
- **Build:** a few hours.

## Not proposed
More members, ViTs, DANN/AdaBN or other domain alignment, label re-basing, and transductive or held-out-draw use. Each is
in declined.md or is the saturated lever that M1 measures.

## Order
M1 + M4 (CPU, same day) -> M2 (CPU) -> M3 (~8 GPU-h) -> ideas 1 + 3 as one bundled arm (~5 GPU-h) -> idea 2 if M3 shows
starved markers -> idea 4 at the end. The total stays well under 25 GPU-h. If M1 + M2 + M3 all say ceiling < .70, report
the ceiling-normalised score and ask the user to re-set the target, rather than running more predictor attempts.
