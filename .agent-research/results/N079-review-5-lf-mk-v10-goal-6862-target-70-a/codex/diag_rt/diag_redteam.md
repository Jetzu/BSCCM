N079 red-team diagnosis — 2026-11-23

**Verdict: the correlations reproduce; joint-profile accuracy and an information ceiling do not follow.** Findings concern interpretation, not broken score arithmetic. No training or repo changes made.

**1. HIGH — Pooled-channel agreement does not establish a jointly correct profile.**

Rebuilt the specified ensemble, checked global-row alignment across all 11 constituent files per direction, and reused the exact seed-0, five-fold FOV OLS:

| Direction; cells | Head: lymphoid / CD45 / CD16 | Composed singles |
|---|---|---|
| b10; 1,191 | .811731 / .778763 / .867603 | .803575 / .795552 / .857061 |
| b01; 2,807 | .794798 / .659679 / .784203 | .797252 / .664444 / .749777 |

Means reproduce **.782796 vs .777944**; all-eight singles **.802762**, head-plus-singles **.810363**.

Controls, using the same FOV folds:

- **Cell-type only, b01:** one-hot `C3` from `n065_tpfov_z2_b01`; predict training-fold pooled means for lymph/gran/mono. Scores **.774035/.642419/.773237**, mean **.729897**, versus singles **.737158**. All cells of a class receive identical predictions. This nearly matches the difficult direction without any within-class profile.
- **Known-class oracle, b10:** on the same 1,091 `cls3>=0` cells, cross-fitted class means score **.845588/.876458/.954855**, versus singles **.817635/.811488/.865884** and heads **.827999/.787908/.888856**. These fluorescence-derived gates are an oracle, not an LF-only predictor. Draw 1 has no true gates; the b10 C3 head lacks source-draw class supervision and is not a sound negative control.
- **Independent encoders:** select CD3 from `n048_v9_base`, CD19 from `n048_v9_zoom2`, CD56 from `n053_cytoland`. No encoder is shared among selected outputs. Lymphoid composed r **.787426/.774909**, versus ensemble **.803575/.797252**. Close, not equal; the control also loses ensembling per marker. These are independently trained multitask models, not newly trained single-task models.
- **Independent within-class shuffles:** permute each single column separately within true b10 classes, 200 times, seed 7901. Mean composed r **.700621/.788481/.828172**. Shuffling loses signal, especially lymphoid signal; the diagnostic is not entirely vacuous, but substantial pooled r survives destroyed within-class identity/dependence.

Four of six headline entries compare singleton CD45/CD16 outputs; they test no multivariate relationship. The remaining channel observes one projection of CD3/CD19/CD56. For weights a,b,c, changes `(delta,-a*delta/b,0)` preserve that projection while changing co-expression. Independent conditional means can compose correctly without recovering a joint distribution. Supported: **single outputs retain pooled-channel predictive information after calibration**. Unsupported: “joint coherence is not the bottleneck.” Held-out-draw OLS is disclosed diagnostic calibration, not inductive performance or physical mixing on raw intensities.

**2. HIGH — Data scarcity and unreachable noise are not established ceilings.**

Re-aggregated K033 (`N040-*/n040_lc_per_target.csv`): 25/50/100% labels yield **.603644/.627158/.642549**; weak4 **.347437/.395465/.430270**. But `N040-*/jobs.txt` fixes **10 epochs** while `--max-train` changes sample count. `scripts/train_lf.py` sets scheduler steps to `epochs * len(dtr)`: doubling labels also doubles gradient updates. This confounds data quantity with optimisation, and concerns v5/older single models, not the current v10 ensemble.

Five failed interventions establish their own limits, not extraction of all useful information. N078 is pending; falling SSL loss neither proves downstream improvement nor excludes it. Stronger representations can improve extraction from existing observations. K043 explicitly says which background label is wrong is unidentified and **no ceiling number follows**. Slide disagreement supports readout/domain-shift concerns, not an unreachable error floor.

**3. MEDIUM — DeepIFC is not a BSCCM ceiling or quantified headroom.**

The cited marker r values are correctly transcribed. However, 527,107 is total data: **247,107 training**, 80,000 validation, 200,000 test cells. It uses frozen PBMC, donor-held-out evaluation, 60× imaging at .33 µm/pixel, compensated fluorescence images, and **a separate network per marker**. These differ from BSCCM's population, optics and supervision. [DeepIFC, Methods](https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.24770).

Its code averages fluorescence images; BSCCM scores v10 transformed targets. Pearson r is not invariant to nonlinear asinh transforms. Without harmonised targets/populations, neither a “~.6 ceiling” nor a numerical CD19 improvement budget follows. [Author evaluation code](https://github.com/timonenv/DeepIFC/blob/master/celltyping_fulldata.py).

**4. MEDIUM — S-BIAD452 transfer remains an untested bet.**

Its 60× flow BF/laser-SSC differs from BSCCM's 20× angular-LED imaging; compensated stain scales also differ. Only **five of eight markers overlap**, missing CD16/CD123/HLA-DR. The cited table labels 2,214 and 1,318 cells as NKT subsets: “3.5k NK cells” is not established by those counts. [Source paper, acquisition/panel/counts](https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.23920).

A second stem does not itself align morphology or label scales. Even DeepIFC's frozen-to-fresh shift reduced CD56 r **.50 to .17**. [DeepIFC §2.4](https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.24770). Panel completeness does not establish transfer probability. Failure would not uniquely identify modality gap: training balance, alignment and optimisation remain alternatives. No transfer experiment was run here.

Reproduction: `/tmp/n079_rt.py`, `/tmp/n079_rt_independent.py`, `/tmp/n079_rt_matched.py`; results `/tmp/n079_rt_{results,perm,independent}.csv`, `/tmp/n079_rt_extra.json`. Run with `/tmp/n079-redteam-venv/bin/python -I`. Container FUSE failed; dependencies/scratch stayed in `/tmp`.
