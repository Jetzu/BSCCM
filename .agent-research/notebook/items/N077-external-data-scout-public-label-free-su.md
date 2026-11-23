# N077 · External data scout: public label-free + surface-marker (or label-free-only blood-cell) datasets that could train/pretrain the LF->marker model; user allowed other data 2026-11-23

type: analysis · date: 2026-11-23 · status: open
goal_effect: n/a · scale: ? · value: - · why: measurement

## Goal
USER 2026-11-23: other datasets allowed if they could help LF->marker decoding (multi-marker decoding is the priority).

## Plan
<!-- exp: hypothesis + expected result · several changes allowed · baseline = current best · kill criterion -->

## Changes
Scout agent (data_scout.md, every claim URL + quote): ranked public datasets. Verified first-hand (study.json/filelist.json from the
BioStudies API): S-BIAD452 "Single-cell imaging dataset of healthy human white blood cells" (Lippeveld et al., Cytometry A 2020,
PMID 31688997), ImageStream-X MK II, whole blood, ONE 9-antibody cocktail on every cell: CD15, Siglec8, CD14, CD19, CD3, CD45, CD4,
CD56, CD8; per-cell zarr stores hold 12 channels "BF1, CD15, Siglec8, CD14, CD19, SSC, CD3, CD45, BF2, CD4, CD56, CD8" (2 brightfield +
side-scatter darkfield = label-free; 9 marker fluorescence images); 11 zarr.zip stores (~17 GB) + IDEAS per-cell feature .fcs. Licence:
BioImage Archive (CC0/CC-BY policy; record states none) -- academic use OK. Overlap with BSCCM targets: CD45, CD3, CD19, CD56, CD14;
missing CD16, CD123, HLA-DR. Only public label-free leukocyte set with ALL markers on every cell -> directly serves joint decoding.
Others (data_scout.md): BSCCM-coherent (4,304 'all'-slide cells x 566 single-LED images; illumination study), DeepIFC (reference
ceiling CD19 r .61 / CD56 .53 at 60x; data not public), BBBC045 (classes only), Cebinelli FACSDiscover (unverified image content),
Guck deformability (SSL only), OMIP-069 flow panel (joint-distribution prior, no images).

## Evidence
<!-- exp: each run = job id, config, seed, output dir · results vs baseline -->

## Outcome
<!-- on close: bin/effect.sh N077 improved|flat|worse --value V | --void  (n/a --why infra|measurement) [--scale knob|module|paradigm] -->

## Outputs
<!-- files in .agent-research/results/N077-slug/ + external absolute paths -->

## Next
Download S-BIAD452 zarr.zip + IDEAS .fcs to /hnvme/workspace/v181bb10-bsccm/external/sbiad452 (started 2026-11-23); then an exp item:
two-stem multi-task model (BSCCM LED stem + IFC BF/SSC stem, shared trunk + shared marker heads for the 5 overlapping markers + IFC-only
heads) scored on the BSCCM GOAL + MM3. Parked: BSCCM-coherent illumination study (CPU), OMIP-069 joint prior.
