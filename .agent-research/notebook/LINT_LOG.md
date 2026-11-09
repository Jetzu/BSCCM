# Lint log

## 2026-11-08 (through N011)
- fixed: 7 cited-path problems (glob citations results/N00x-*/file expanded to real folder names in N001,N003-N009 items; check.sh mis-parsed the globs as /file)
- merged 0, superseded 0, verified 0, stale 0, promoted 0.
- KNOWLEDGE bullets carry no K ids / verification tags yet; Headline already <=10 bullets, left as is. No unresolvable items.

## 2026-11-09 (through N023; N022/N023 open and left untouched)
- fixed: 8 cited-path problems (glob/brace citations results/N0xx-*/file expanded to real folders in N007, N010, N012, N013, N018, N019, N021); 20 >5MB npz (GS-student outputs of N019/N020/N021) moved to large_output_location /hnvme/workspace/v181bb10-bsccm/agent-research-large (byte-identical copy checked with cmp, symlinked back, absolute path cited in N019/N020/N021); large_output_location set in META.md; commit hashes filled for 12 done INDEX rows (from git log of each item file); resolved TODO "GOAL.md: init" entry removed.
- NOT fixed: 8 >5MB npz in results/N022-* (item open, caller asked not to touch it) -> check.sh still prints these; move them when N022 closes.
- KNOWLEDGE: K ids + tags assigned (K001-K018); Headline rewritten to 10 verified bullets (added N010, N014-N021 findings); non-headline bullets moved to new Topics subsections.
- merged 0, superseded 1 (K004 GOAL best .4835 -> K008/K010), verified 16 (numbers re-read from cited results files), partial 1 (K018), stale 1 (K006: autofl truth not detrended), refuted +6 (N009 PRH rotation, N010/N011 Macenko/DANN, N012 within-slide autofl biology, N013 v3 CD19/CD56 collapse, N014 SSL-useless, N020 unconditional DANN), promoted 2 rules (detrended-labels-only [permanent]; disjoint/pre-registered external evals). Rules dropped: none.
- Follow-ups to TODO: close stale-open items N002-N004, N008, N009, N011, N012; re-check K013/GOAL after targets_v5 (N023).
