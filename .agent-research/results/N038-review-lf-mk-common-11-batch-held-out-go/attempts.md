# Attempts since the last GOAL improvement (N026 ensemble .6515 -> N031 inductive re-base .6527; stall counted since N023)

GOAL = LF->MK common-11, batch-held-out (train one blood draw, test the other), mean of b1to0/b0to1, v5 labels; best .6527 (N031);
noise .002 (flat zone +-.004); target .67 (Claude proposal, unconfirmed).

| item | scale | what changed | GOAL value vs noise |
|------|-------|--------------|---------------------|
| N027 | module | Giemsa STUDENT capacity (DinoBloom-B 8 blocks vs -L) on the TT6 chain; clinical/autofl readouts | void (GOAL not measured; worse on its own metric) |
| N030 | module | Giemsa student with FEWER trainable blocks (frozen 10/12, fully frozen) | void (GOAL not measured; flat on its own) |
| N033 | module | virtual-stain data engine: Giemsa student trained also on 318k virtual Wright images with measured labels | void (GOAL not measured; flat) |
| N034 | paradigm | LF -> fluorescence IMAGES by conditional flow matching + set-level distribution matching bench | void (GOAL not measured; distributionally better, per cell worse; confirmed N035) |
| N037 | module | GOAL recipe retrained on cross-fitted, Giemsa-free-calibrated labels (v5cx) | .6537 vs .6527 = +.0010 (flat, < .004) |

Observation: only ONE of the five (N037) touched the GOAL predictor at all; N027/N030/N033 optimised the downstream Giemsa student and
N034 changed the output (images / sets). The GOAL predictor itself (SSL(aug)-init ResNet-50, 128-px LF crop, per-cell regression,
one draw for training, base + 2x-zoom ensemble) has been frozen since N026/N031.

**All 5 attempts assumed the per-cell LF->marker predictor (SSL-init ResNet-50 on single 128-px LED crops, one training draw) is
already near what the data allow, so the remaining gap had to be closed AROUND it -- by cleaner labels, a better Giemsa student, or a
different output form -- rather than by changing what the predictor sees or how it is trained.**
