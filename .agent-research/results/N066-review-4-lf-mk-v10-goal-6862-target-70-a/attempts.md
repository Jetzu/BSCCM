# N066 look-back: 5 flat attempts since review N059 (GOAL best .6862 N053, bar .6916 = best + 2 x noise .0027)

| item | scale | what changed | GOAL value | vs bar |
|---|---|---|---|---|
| N060 | module | rare-positive output heads (HL-Gauss, two-part gate x amplitude), r50 recipe, 2 seeds; single models vs control .6706 | twopart .6756 / hlgauss .6591 (single) | single-model, no ensemble gain |
| N061 | module | within-FOV pair loss, per-FOV random-effect offsets, autofl removed from shared encoder, Pearson loss | .6742 / .6740 / .6713 / .6726 (single) | flat |
| N062 | paradigm | panel-matrix latent head (shared molecular factors for single + pooled assays, assay decoders, per-slide adapters) | .6704 (single; pooled groups +.014..+.031, singles -.01..-.02) | flat |
| N063 | module | composite: pooled groups from panel head, singles from twopart + fov-re; fresh seeds 2-3 | single .6784 vs .6678 control (+.0106); as Cytoland-ens member .6902 | flat (ensemble) |
| N065 | module | every ensemble member rebuilt on the composite recipe (base, zoom2, Cytoland); pre-registered 3-member E3c | E3c .6899; descriptive 6-way (E3c + N053 members) .6929 | flat (.6899 < .6916) |

Per direction (6-way): b10 (train draw 1 -> test draw 0) .7036 = already above .70; b01 (train draw 0 -> test draw 1) .6822.
b10 - b01 per target: grp:CD45 +.119, grp:CD16 +.083, HLA-DR +.080, CD123 +.038, CD3 +.036; reverse: CD45 -.100, CD56 -.087.
Training labels per draw: pooled groups 10,210 (draw 0, ONE 'all' slide) vs 27,074 (draw 1); CD3 6,538 vs 31,332; CD16 9,397 vs 27,857.

**All 5 attempts assumed the remaining gap is a modelling problem of the readout (head, loss, ensemble member) on a FIXED training
set -- the same labelled cells, one training draw, the same v10 targets used both as training signal and per-target independent
outputs -- and that a better function of the image of the SAME cells, trained the same way, can close it.**
