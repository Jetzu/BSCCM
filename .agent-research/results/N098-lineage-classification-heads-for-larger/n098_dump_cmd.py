"""Print the train_lf.py command that re-creates a finished run's model from its result.json args, plus --dump-features 1
(train_lf.py refuses unless the rebuilt config and rows equal the checkpoint's)."""
import json, shlex, sys
run = sys.argv[1]; a = json.load(open(f'{run}/result.json'))['args']
cmd = ['scripts/train_lf.py', '--out', run]
for k, v in a.items():
    if k in ('out', 'dump_features', 'dump_train_preds', 'adabn') or v is None: continue
    f = '--' + k.replace('_', '-')
    if isinstance(v, bool):
        if v: cmd.append(f)
        continue
    if v == '': continue
    cmd += [f, str(v)]
print(' '.join(shlex.quote(c) for c in cmd + ['--dump-features', '1']))
