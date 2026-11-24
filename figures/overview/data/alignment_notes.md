# Alignment metrics: comparability notes

Numbers in parentheses are line numbers in `alignment_metrics.csv` (header = line 1).

## LF<->GS cross-modal
- **Comparable:** cross-slide exact-cell R@k on the full held-out GS slide (b1: 5,697 cells; b0: 10,015 cells) in N009, N010, N045 v2 and N054. LF encoders and training budgets differ, so these compare recipes.
- **Not comparable:**
  - FOV-split retrieval. N009 pools both slides; N045 splits within each slide.
  - N028 top-k among 1,000 candidates.
  - Autofluorescence r, whose truth and readout change between items: v3 (N009/N010), v4 (N010 l.17), v4c within-slide (N028), v5c ridge head (N045), student on clean labels (N054).
  - Only the TT6c student lines up across items: N045 (529-530) = N054 (593/575).

## GS->clinical
- **PBC contrast sets differ:** 8+2 (N020/N021), 12 (N045), and PBC10 (N056-N058).
- **MLL23:** the "fine mean of 12" in N022/N033 uses the same contrasts as N056 `MLL23_primary12`, but with different teachers. `MLL23_15` (N056-N058) adds 3 lineage references. N054 M6 is unanchored, so it is not comparable.
- **Noise:** seed range up to .15 within one teacher (N056) and teacher spread .48-.61 (N032). Compare only the 3-teacher means from the shared N056-N058 harness.

## LF batch shift
N014 (12 markers, v4) and N026 (common-11, v5) are on different scales. Compare the paired deltas only.

## other
N034-B is a scalar LF->MK set-matching bench. N009 MK->LF stands alone.

## Headline comparisons
1. **Cross-slide R@10 tops out at .044 whatever the method (chance .0018).** SigLIP .038 (390), SOTAlign .036 (395), CCA .021 (370), Procrustes .019 (380), N009 SigLIP .033 (42), N010 DANN .033 (85), N054 .044 (646).
2. **Distillation carries markers and shared spaces do not.** Autofluorescence r: SOTAlign .45/.22 (397/432) and SigLIP .38/.18 (392/427), against TT6c .715/.601 (593/575) and hub .751/.549 (605/587). Unpaired GW scores .18/.46 (602/584).
3. **Unconditional DANN on PBC is unstable while class anchoring transfers lineage.** PBC mean is .41 without adaptation and .46/.56 with DANN (157-159). Eosinophil goes .11 -> .955/.356 with anchoring (170/180/190), and adding DANN drops basophil to .151 (205).
4. **MLL23_15: the hub target matters more than anchor strength, and post-hoc alignment does about nothing.** TT6c .568 (766); anchor 2/8: .604/.627 (910/916); matched hub .640/.681 (722/930); hub .704 (678); T+OT/mean/CORAL .562/.571/.571 (789/803/775).
5. **LF batch adaptation is flat or worse.** DANN .5541 vs .5545 (124/121); AdaBN -.005 to -.011 (259-266).
