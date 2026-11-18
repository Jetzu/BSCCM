# Attempts since the last review (N047) -- v10 GOAL (LF->MK common-11, batch-held-out r), noise sd .0027
| item | scale | what changed | value (v10) | vs best |
| N048 | module | labels: asinh cofactor 150 (training labels v9) | .6804 (ens) | improved (after v10 re-base) |
| N049 | module | noise-aware positive/negative labels | n/a | falsified at fitting |
| N050 | module | heterogeneous ConvNeXt-T member | .6533 (v5) | flat |
| N051 | fix | v10 labels re-base (user decision) | re-scored N031 .6703, N048 .6804 | re-base |
| N053 | module | pretrained encoders (DINOv3 S/B, DinoBloom-B, Cytoland) + per-LED gate; Cytoland as 3rd ensemble member | .6862 | IMPROVED (+.0058) |
| N054 | paradigm | tri-modal latent designs (two-space + flow vs hub vs TT6c) | void (Giemsa metrics) | flat |
| N055 | module | more Cytoland members (zoom2, cofactor-150 labels) | .6902 (pre-reg 4-member) | flat (+.0040 < .0054) |
| N056 | paradigm | clinical transfer: hub vs TT6c anchored students + OT/CORAL/mean-shift | void | flat (seed floor) |
| N057 | module | TT6c anchor-strength sweep .5 -> 2, 8 | void | flat (seed floor) |
GOAL-relevant attempts since N053: N055 only (+.0040); N054/N056/N057 are Giemsa/clinical (void for the GOAL).
Best single models: SSL r50 base .6730; Cytoland .6671; ensemble members add ~.002 each. Best ensemble cross-draw .6862 equals the
in-distribution single-model reference (.6862, N051).

All k attempts assumed that the remaining LF->MK error is predictor error that a better/more diverse image model can remove, with the
v10 labels of the held-out draw treated as a fixed, reachable truth.
