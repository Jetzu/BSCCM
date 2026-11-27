# N095 look-back: 5 flat attempts since N085 (record GOAL .6967; target .70; noise .0027; improved = > .7021)

| item | scale | what changed | value | vs noise |
|---|---|---|---|---|
| N087 | module | decoders on frozen features (rank-k, MLP, kNN, flow), image autoencoders, learned marker priors (sandbox) | .6700 (best decoder; own head .6730) | below the single-model reference; flat |
| N090 | knob | feature-preserving fine-tuning (encoder lr x0.1/x0.3, LP-FT/WiSE-FT) | .6617 vs control .6706 | worse (-.009 = 3.3x noise) |
| N091 | paradigm | DINO self-distillation SSL encoder as a 9th ensemble member | E9 .6951 vs E8 .6967 | flat (-.0016) |
| N093 | paradigm | class-conditional Giemsa -> marker matching (clinical harness, not GOAL) | void (MLL23 .494 vs distillation .66) | worse on its own metric |
| N094 | knob | every E8 member averaged over 2 seeds (8 new seed-5 runs x 2 directions) | .6973 (seed-5 replicate .6961) | flat (+.0006) |

Before these (still relevant): N055 ensembling saturates near .69 (+.002/member); N084 4th member .6911; N085 record = 8 members.
Where the score sits (K043, N066): b10 .7066 already above .70; the deficit is b01 (.6869): grp:CD45 b01 ~.66, HLA-DR ~.45,
grp:CD16 ~.78; within the b01 test draw grp:CD45 scores .881 on 'all' replicate 1 vs .578 on replicate 0. Residual FOV-level
clustering above a permutation null (CD19 +.43-.46, CD56 +.29); FOV-level label error caps per-cell r (K028).

**All 5 attempts assumed the per-cell label is a fixed, correct target and that the remaining gap lives in the image->prediction
mapping (better representation, decoder, fine-tuning or more/averaged ensemble members), not in the draw-/slide-/FOV-specific
label and acquisition shift of the b01 test draw and not in how the 13 outputs relate to each other (each target is fitted and
scored independently).**
