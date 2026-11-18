# N059 ideas: synthesis and chosen portfolio
Sources: agents/outsider.md, agents/assumption_breaker.md, agents/codex_ideas.md, agents/codex_diag_redteam.md (+ literature
agents lit_crossbatch.md / lit_noisy_ceiling.md appended when they land). Diagnosis (diagnosis.md, incl. corrections) reproduced.

## What the review established (all reproduced first-hand)
1. Ensembling is closed: test-selected greedy ensemble over 22 members peaks .6922, r_inf fit .693 (M1).
2. The cross-draw shift is real for matched single models (n023_v5 vs in-dist: -.026, AB s3) and concentrates in HLA-DR and the
   pooled panel, but its causes sit in the HELD-OUT LABELS: draw-1 HLA-DR stain near floor and its label tracks the same cell's
   autofluorescence (lymphocytes r -.606 vs +.093 in draw 0; AB s7); the grp:CD45 b01 deficit lives on one dim slide (rep0 .572 vs
   in-dist .656; rep1 .876 vs .869; AB s3) -> per-slide, not per-draw, and not learnable inductively.
3. Inductive upside on those targets <= ~+.01 (AB); monotone recalibration oracle +.0045; few-shot cross-output readout fitted on
   held-out labels reaches .7042 (+.018) but only with >= ~200-500 labelled new-draw cells per target (AB s3 f).
4. Weak markers are few-shot rare-positive problems: CD123 top-1 % cells hold .88-.92 of test variance (M4); CD123 carried 59 % of
   the N053 gain (.0034/.0058) and has seed spread up to .124 (AB s8) -> GOAL noise .0027 understates per-item risk.
5. FOV-level label error still costs ~.04 (within-FOV .726), concentrated in CD19/CD56.

## Ideas, deduplicated and ranked (P(success) x gain / cost)
| # | idea | sources | scale | expected | cost | decision |
| A | Rare-positive output heads for dim markers: HL-Gauss distributional head (E[y|x] in label space) + two-part gate x amplitude head; CD123 bagged over seeds | outsider 1+3, Codex 4, AB 3 | module | +.003-.008 | ~2.5 GPU-h | N060 |
| B | Within-FOV pairwise-difference loss (FOV offsets cancel) + drop co-stained autofl supervision from the shared encoder | Codex 1+3 | module | +.004-.007 | ~2.5 GPU-h | N061 |
| C | Panel-matrix latent head: shared molecular factors across single + pooled assays with assay decoders, per-slide adapters for pooled training slides | outsider 2, Codex 2, AB 4 | paradigm | +.003-.01 if markers are label-starved | ~2 GPU-h + 1-2 d build | N062 |
| D | Few-shot new-draw readout (protocol change: uses held-out labels) | AB 1 | protocol | .70-.705 at ~500 cells/target | 0 GPU-h | USER DECISION (analysis only, not a GOAL attempt) |
| E | Fluorescence spatial descriptors as privileged auxiliary targets | Codex 5 | module | +.002-.006 | 2-3 d build | parked (TODO) |
| F | Split-half reliability r_max per marker from raw fluorescence (M2) | outsider M2 | diagnostic | ceiling estimate | CPU hours | parked (TODO; run if portfolio flat) |
| G | Learning curve 1/8..1 of the source draw (M3) | outsider M3 | diagnostic | starvation test | 6-10 GPU-h | parked (cheaper partial: K033 already +.035 on weak targets 50->100 %) |
| H | Mask / down-weight the dim rep0 'all' slide in b1to0 training | AB 2 | knob | 0-+.004 (class-mean proxy flat) | .4 GPU-h | folded into C (per-slide adapters) |
| I | Per-target ensemble weights / cross-target stacker on source OOF | outsider 4 | knob | +.002-.005 | 0 GPU-h | parked (needs source OOF preds; cross-target inputs may count as lineage priors) |

## Chosen portfolio (3 exp items, ~7 GPU-h total, < 25 -> no approval needed)
- N060 (module) = A, N061 (module) = B, N062 (paradigm) = C. Writer: Codex (one brief, three flags in the GOAL harness), Claude red-team
  + integration. Each arm screened with 2 seeds per direction (per-item noise, AB s8), scored on the frozen v10 GOAL and as a 4th
  ensemble member; pre-registered bar: single-model mean over 2 seeds > r50 base (.6730) + .0054, or ensemble > .6862 + .0054.
- Re-scoping options for the user (review #3 without progress beyond ensembling): (1) allow few-shot new-draw readout (D) as a
  second, clearly labelled protocol; (2) keep the inductive GOAL and accept ~.69-.696 as the two-draw ceiling; (3) download the
  BSCCM-coherent set (user OK'd, low expected value) to separate draw from slide effects.

## Literature (agents/lit_crossbatch.md; lit_noisy_ceiling.md pending)
No published inductive fix for a batch-specific fluorescence READOUT: CytoNorm, cyCombine, gaussNorm, CytoVI, AutoSpectral all need
data from the new batch (controls or the batch's own marker data); label-free -> marker papers never vary the readout across their split
(DeepIFC held-out donors share one panel/compensation: CD19 r .61, CD56 .53 on PBMC). BSCCM paper: batch-1 pooled slides got 35 % CD19
antibody; authors call the CD123/CD14/HLA-DR channel "too noisy". Verified first-hand earlier (N038: DeepIFC numbers, 35 % CD19).
Folded in: crowd-layer mixed-effects training -> N061 arm (i); measurement-model pooled heads -> N062. Simulated readout randomisation
(lit idea 3) parked: CPU hedge test on saved predictions first (TODO).
lit_noisy_ceiling.md (landed after close): Pearson-optimal predictor under label noise = conditional mean (MSE already); controlled
evidence on noisy held-out labels: MSE + Pearson loss weak yes (Visium HD PCC .293 -> .325, overlapping spreads), HL-Gauss yes for RMSE,
tail/imbalance reweighting and rank regularisers NO (STS-B reproduction: vanilla .766 > LDS .762 / FDS .751 / RankSim .756 / ConR .735),
heteroscedastic Gaussian heads no, self-distillation no gain at ensemble level, local neighbour context yes in spatial transcriptomics
(+.016). Folded in: N060 gets a 3rd arm (MSE + .5 x per-target batch Pearson). Added to TODO: noise-ceiling bracket (unstained floor +
pixel split-half + FOV residual variance; decides reachability) and neighbour-bleed residual test (CPU).
Neighbour-bleed falsifier run (n059_neighbour.py -> n059_neighbour.csv, CPU): nearest neighbour median 262-380 px (crop 128 px);
partial r of ensemble residual with proximity x neighbour prediction mean |.023|, max |.066| (CD19 b10) -> KILLED (< .05 bar).
