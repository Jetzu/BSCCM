# N091 red-team (Claude) of codex/train_ssl_lf.py

**Verdict: NOTHING BREAKS** (no crash, leak or resume bug). One claim is wrong, and there are small departures from DINO.

## Findings
- **MAJOR (the claim is false): the last layer is not frozen in epoch 1.** Line 328 freezes only the magnitude g. The directions v train in epoch 1 (measured grad norm 0.011). From epoch 2, g trains and is weight-decayed. DINO's default is the reverse: g fixed at 1 all run, all last-layer gradients cancelled in epoch 1. DINO calls a free g potentially unstable. Codex followed the ambiguous brief (README:25); decide before the runs.
- **MINOR: weight decay applies to every parameter.** The .04→.4 decay also hits BN, biases and g (263-264, 342-343). DINO exempts biases and 1-D parameters. Summed lr×wd is 2.94, against 1.07 for SIGReg.
- **MINOR: the prototype logits are bf16.** Autocast undoes the `.float()` at line 35. Softmax and temperatures run in fp32 (70-76). Measured target KL ≤1e-4 (fp32 vs bf16): negligible.
- **MINOR: `teacher_kl_uniform` adds no information.** It equals log K − entropy (357). `teacher_marginal_kl_uniform` (395-400) is correct and is the useful collapse signal.
- **MINOR (inactive by default): local crops are not independent views.** Each crop shares its source global view's augmentation (348), yet is paired with that view's teacher output (77-78).
- **MINOR (outside the claim):** `train_lf.py:353` `--bottleneck-init projector` raises KeyError on a DINO checkpoint.

## Verified correct
- **(a) Loss:** the teacher sees only the global views (344-348). Pairs with i=j are skipped. Softmax is over prototypes. Centring is teacher-only. The centre comes from un-centred logits (74) and is updated after the step (370). τs=.1. The τt warm-up matches DINO's linspace over 12 epochs.
- **(b) EMA:** runs after `opt.step` and covers the encoder plus the head; momentum is cosine .996→1; no teacher gradient. The teacher's BN stays in train mode (its own running statistics, not EMA-averaged), which is standard: DINO never calls `teacher.eval()`.
- **(e) Inputs:** as SIGReg.
- **(f) Wall time:** plausible. The SIGReg baselines (1295538 ex0, 1:52 on h12-02 at ~1.9k cells/s; 1295541 ex1, 0:36 on h12-09 at ~2.8k cells/s) were each multiplied by 1.3. The ex0 figure is pessimistic (slow node): about 1.5 h at the ex1 rate. Total ≤3.2 H100-h.
- **(g) Codex's tests:** 5/5 pass (paths-only copy, /tmp/n091_rt). SIGReg losses are identical across old, new and legacy-resume runs. DINO resume gives 5.712509632110596 both ways.
