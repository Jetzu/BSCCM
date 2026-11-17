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

## 2026-11-12 (through N036; N035/N036 open and left untouched)
- fixed 0: check.sh printed nothing (N022 npz issue from last lint already gone). Misplaced K026/K027 (were appended under Refuted) moved to new Topics subsection "Generative / set-level matching". Stale TODO entries removed (INDEX items N002-N012 now done; N023 re-check done, K013 already [partial]).
- merged 0, superseded 0, verified 0 (no [unverified] bullets), stale 0, promoted 0 (rules unchanged, 2 lines, none dropped).
- Headline rewritten from 15 to 10 verified bullets (K019 K025 K023 K020 K011 K012 K016 K001 K007 K008); K010/K013/K014/K015/K021/K022/K024 moved to Topics "Clinical transfer and LF teacher". All K001-K027 retained.
- Not covered: N034 K026/K027 are [partial] 1-seed; N035 result pending (open); N036 open. INDEX.md/N035 item/N036 files carry uncommitted edits of a parallel session and were NOT staged.

## 2026-11-18 (through N051; N044 open and parallel N052+ left untouched)
- fixed 1: check.sh printed nothing; INDEX N034 row had its commit hash in the outcome column -> moved to commit column (outcome text added). Other done rows already carry hashes.
- KNOWLEDGE: added K031 (N036/N037/N039 label-side fixes, flat/QC-fail), K032 (N041/N042/N050 model-side nulls), K033 (N040/N046 noisy student), K034 (N049 noise model not identifiable + N051 correction), K035 (N045 SOTAlign negative); K031-K035 numbers re-read from results CSVs. Headline rewritten from 13 to 10 bullets (K030 K029 K028 K033 K019 K025 K023 K011 K001 K007); K008/K012/K016/K020 moved to Topics. K023 annotated: v5-metric best, GOAL since re-based (K030).
- merged 0, superseded 0, verified 0 (no [unverified] bullets), stale 0, promoted 2 rules (re-base is not progress; <2 x noise = flat; total 4 lines, none dropped).
- Not covered: N044 (open). K035 [partial] 1-seed screen. N043 figures item has no knowledge bullet (infra).
