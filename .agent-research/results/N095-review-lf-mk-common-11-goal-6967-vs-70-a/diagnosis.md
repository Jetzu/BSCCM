# N095 diagnosis (Claude): where the remaining gap lives

Gap: target .70 - record .6967 = .0033 (1.2 x noise); a NEW record needs > .7021 (the 2 x noise rule), i.e. +.0054.
Score split (N085 E8): b10 (train draw 1 -> test draw 0) .7066, already above .70; b01 (train draw 0 -> test draw 1) .6869.
The whole deficit is b01, concentrated in 3 targets (K043, N066 frozen ensemble): grp:CD45 ~.66, HLA-DR ~.45, grp:CD16 ~.78.

## Bottleneck hypothesis
The limit is the TEST-DRAW-SPECIFIC LABEL / ACQUISITION SHIFT of draw 1 (slide-, replicate- and FOV-level label offsets from the
fluorescence background model), not the image representation, decoder, or ensemble variance.

## Evidence
1. Representation/decoder/ensemble levers are exhausted: N087 no decoder beats the own linear head; N053 no encoder beats SSL r50
   alone; N055/N091 extra members +.002 or less; N094 seed-averaging +.0006 and an independent full re-draw of member seeds .6961
   (ensemble variance is not the limit); N090 gentler fine-tuning worse.
2. Same model, same draw, different replicate slide: grp:CD45 .881 on 'all' replicate 1 vs .578 on replicate 0 (K043; survives
   label-SD matching .81 vs .49) -> the per-slide label/acquisition context, not the cells, decides the score.
3. Residual FOV clustering above a permutation null (CD19 +.43, CD56 +.29, grp:CD45 b01 +.21) that LF context cannot predict
   across draws (K043); within-FOV scoring was .7211 vs pooled .6527 in the N031 era (K028) -> FOV-level label error caps r.
4. Alternative background model labels (z(v10)+z(local-corner bg)) give .6996 vs noise-matched control .6722, gain FOV-level (K043).
5. N096 quick screen (11-27, 15 % subset): label correlations between co-measured targets CHANGE SIGN between draws
   (single:CD3 ~ autofl -.31 in draw 0 vs +.34 in draw 1; single:CD45 ~ autofl .09 vs .60; single:HLA-DR ~ autofl .37 vs -.21)
   -> the labels carry draw-specific structure that no image model can learn from the other draw.
6. Censored (clip-floor) label fractions differ between draws (CD3 .094/.427, grp:CD16 .307/.082, grp:CD45 .095/.179; K044)
   and MSE treats them as exact.

## Ceiling instruments already measured
- Within-FOV oracle (remove FOV means): .7211 vs .6527 (N031 era, v5 labels) -> about +.07 of r sits in FOV-level variance.
- Per-slide: grp:CD45 .881 vs .578 within the same draw.
- Type oracle on ImageStream: macro r .814 vs .596 (cell-type knowledge alone carries most of the signal there).
Not yet measured on the record: (a) per-slide / per-replicate score of E8 b01 with slide-level label offsets removed (oracle per-slide
mean + SD), (b) how much of the b01 deficit closes if HLA-DR / grp:CD45 / grp:CD16 took their b10 values.

## What would falsify it
- If an oracle that removes per-slide (or per-FOV) label means/scales from the b01 test labels does NOT raise E8 b01 above .70,
  the label shift is not the bottleneck (then per-cell biology within type is).
- If a model trained with the alternative background labels or censoring-aware loss raises b01 on the UNCHANGED v10 labels by
  > .0054, the bottleneck was in how training labels are used, not in the test labels.
- If training-free per-slide standardisation of predictions (uses only test inputs; label definition is per-slide z for
  multi-slide targets) moves GOAL by << .005, slide-level offsets in predictions are not the issue.
