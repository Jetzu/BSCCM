# N079 look-back: 5 flat attempts since review N066 (GOAL best .6862 N053, bar .6916; single-model control .6706)

| item | scale | what changed | GOAL (single, 2 seeds) | vs control .6706 |
|---|---|---|---|---|
| N067 | module | training targets = consensus of v10 + local-corner-background label versions (eval v10) | .6624 | -.0083 WORSE (CD56 .504 -> .444) |
| N068 | module | exact-symmetry augmentation (anti-diagonal reflection + LED channel swap) + 2-view TTA; LED119 dropped | .6694 / drop-only .6718 | flat |
| N073 | module | clip-floor labels as upper bounds (one-sided loss) + 20 epochs | .6658 | -.0048 flat |
| N075 | module | second head on the local-background label version (separation, Wei et al.) | .6708 | +.0002 flat |
| N076 | module | final fine-tune only on 'all'-slide multi-marker cells (user priority) | .6681; MM3 .7564 vs .7577 | flat (MM3 -.0012, bar +.007) |
Pending (feed into this review): N074 dense fluorescence-image supervision (paradigm), N078 in-domain SSL 4x longer / ConvNeXt-S.

**All 5 attempts assumed that re-processing the SAME BSCCM measurements -- other label versions, censoring, symmetries, sample
weighting toward the multi-marker slides -- can put more information about each cell's markers into the model; none added a new
measurement (new labelled cells, new joint marker observations, a richer per-cell signal, or a stronger image representation).**
