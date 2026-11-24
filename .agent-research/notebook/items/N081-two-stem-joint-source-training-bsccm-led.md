# N081 · Two-stem joint-source training: BSCCM LED stem + S-BIAD452 BF/SSC stem, shared trunk + shared marker heads with per-dataset adapters; BSCCM GOAL/MM3 + IFC held-out-donor joint metrics

type: exp · date: 2026-11-23 · status: running
goal_effect: pending · scale: module

## Goal
Review N079 idea #2 (Codex 1, lit B): add NEW joint observations to BSCCM training -- S-BIAD452 cells with all 9 markers (5 overlap with
BSCCM single targets: CD14, CD19, CD3, CD45, CD56) through a second input stem on the shared trunk.

## Plan
Codex writes the harness extension (--ext-ifc/--ext-w/--ext-test-donor; IFC conv1 stem -> shared r50 trunk; shared reg rows + per-dataset
affine for the 5 overlapping markers; IFC-only head for CD15/Siglec8/CD4/CD8; one IFC batch per BSCCM step; BSCCM eval unchanged; extra
IFC held-out-donor eval). Claude red-team + flag-off equality suite before GPU. Screen: GOAL r50 recipe + --ext-ifc, ext-w .5, seed 0,
both directions vs control n051_v10_base seed 0. PRIMARY (pre-registered): GOAL delta > +.0054 OR CD19/CD56/CD3/CD14/CD45 mean delta >=
.02 with GOAL >= -.0027; MM3 reported. ~2 GPU-h.

## Changes
Codex build (codex/, 13 CPU tests, flag-off byte-equal vs the live harness 4cf89b09; red-team re-ran the N074 suite 7/7).
Claude red-team (agents_redteam.md): BLOCKER shared BN running statistics across sources (README states BN params AND running stats are
shared -> eval normalises with a ~.47/.53 BSCCM/IFC mix; red-team CPU repro: BSCCM eval features cosine .88 vs BSCCM-only stats) --
confirmed by the design statement; MAJOR donor naming (packer runs sample1_early/late; exact-name exclusion fails or leaks) -- confirmed
(my packer); MAJOR label-scale mismatch on shared rows -- PARTLY reproduced: v10 SDs CD14 .98/1.00, CD19 .36/.52, CD3 1.00/1.01,
CD45 .98/1.00, CD56 .92/.84 (only CD19 mismatched; red-team's .26/.20/.49 not reproduced) -> fix kept as data-driven init; minors:
packer stats on all donors, shared clip, global RNG, enc-lr group, adabn, augmentation loop.
Fix round: Codex (codex_fix/brief.md): per-source BN buffers, donor-prefix mapping, data-driven affine init + no weight decay, separate
IFC RNG, enc-lr group, reject adabn, vectorised D4, per-source grad-norm logging, packer --stat-donors.
Fix round delivered (codex_fix/: 22/22 CPU tests). Claude verification: read the BN buffer swap (reference swap around the IFC forward,
restored on exceptions, affine weights shared); re-ran all suites locally: test_fix 9/9 (incl. ext_w=0 -> BSCCM BN buffers, RNG, rows and
eval predictions identical to the IFC-skipped control over 3 optimizer steps), test_ifc 6/6, N074 flag-off regression 7/7. Integrated
(diffs applied, cmp-identical to staged copies). Pack rebuilt with --stat-donors sample0,sample1 (test donor sample2) -- job in jobs.txt.

## Evidence

2026-11-24 data recovery: S-BIAD452 sample2 (donor 2) zarr.zip files lack .zarray/.zattrs on the archive (chunk files only) -> packer
rebuilds the array metadata from a metadata-bearing part (chunks 100, blosc-lz4, vlen <u2) and infers each cell's (H, W) from its
flattened length by brightfield row smoothness (2084/2084 exact on sample0 cells with known shapes). Without object numbers donor 2
cannot be linked to IDEAS -> labels are IMAGE-DERIVED for all donors (masked background-subtracted sums; asinh r vs IDEAS on sample0
CD15 1.00, Siglec8 .977, CD14 .948, CD19 .762, CD3 .998, CD45 .997, CD4 .966, CD56 .937, CD8 .908; IDEAS kept as I_ideas). Earlier pack
failures: truncated FCS download (4 KB, re-fetched); all 11 zips CRC ok, uncompressed sizes = archive sizes.
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N081 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N081-slug/ + external absolute paths -->

## Next
