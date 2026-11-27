# N097 · Lineage separation of decoded markers (user 11-27): are predicted CD3 and CD19 exclusive, can the models separate T/B/NK/monocyte/granulocyte lineages (BSCCM single-stain positives + ImageStream sandbox truth); figures

type: analysis · date: 2026-11-27 · status: done
goal_effect: n/a · scale: ? · value: - · why: measurement

## Goal
User 11-27: "I mean CD3 and CD19 must be anticorrelated. are they, can the models separate cell lineages? i want figures on this".

## Plan
Saved predictions only (no training, CPU). BSCCM: lineage positives on each marker's own single-stain slide of the TEST draw (GMM valley;
rare-positive markers also > median + 3 MAD, after the plain valley called 60 % of draw-1 CD19 cells positive); record ensemble E8 +
8 members, both directions. Sandbox: N080 full/single models, gates on the held-out donor's own labels (N080 hierarchical gates).

## Changes
n097_lineage.py (analysis), figures/overview/make_lineage.py (fig11, fig12), CAPTIONS.md, README gallery + N097 entry.

## Evidence
BSCCM, record ensemble (b10 / b01): own-slide AUROC CD3 .993/.864, CD19 .831/.872, CD56 .914/.942, CD14 .983/.973, CD16 .970/.974,
CD123 .867/.862; B vs T by predicted CD19 - CD3 .947 (39 B, 89 T) / .695 (10 B, 692 T); members .918-.950 / .627-.760.
r(pred CD3, pred CD19): all cells +.232/+.181; gated lymphocytes (draw-0 'all' slide) -.535; true T+B -.493/+.006.
b01: 67 of 692 measured-CD3+ cells have predicted CD19 z > 2 (+ higher predicted HLA-DR), spread over 63 of 249 FOVs, CD3 label
median 1.67 -> model error (B-like T cells), not an FOV label artefact.
Sandbox (donors 0/1/2): measured r(CD3,CD19) in lymphocytes -.628/-.881/-.675; predicted all-marker -.931/-.894/-.836, one-marker
-.860/-.934/-.829 (over-separated: one T<->B axis); AUROC B vs T all-marker .738/.940/.459, one-marker .807/.902/.512; predicted vs
measured CD19 r .530/.657/.091 (all-marker) -> donor 2 CD19 not predicted, B vs T at chance there.
Figures: figures/overview/fig11_lineages_bsccm.png, fig12_lineages_sandbox.png (captions in CAPTIONS.md + .json).

## Outcome
Models DO learn CD3-CD19 exclusivity as an output direction (within lymphocytes -.54 on BSCCM; sandbox even more anticorrelated than
the truth) but over all white cells the two outputs correlate positively (both lymphocyte markers). Big lineages (granulocyte CD16,
monocyte CD14, T CD3) are detected at AUROC .86-.99; T vs B works in b10 (.95) and weakly in b01 (.70); the limit is how well CD19
itself is predicted (donor-dependent in the sandbox). Analysis only: n/a (measurement).

## Outputs
n097_lineage.py, bsccm_{lineage_profiles,auroc,cd3_cd19_corr,tb_cells}.csv, sandbox_{cd3_cd19,type_profiles,lymph_cells}.csv; figures/overview/fig11/fig12.

## Next
Add N096 corr-loss sandbox arms to fig12 when job 1471630 finishes (the script picks them up automatically).
