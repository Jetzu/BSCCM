import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

stage = Path.cwd()
repo = next(p for p in stage.parents if (p/'bsccm/data.py').is_file())
source = repo/'scripts/train_lf.py'
patched = stage/'train_lf.py'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
left, right = ast.parse(source.read_text()), ast.parse(patched.read_text())
for name in ('prep', 'fwd', 'evaluate'):
    a = next(n for n in left.body if isinstance(n, ast.FunctionDef) and n.name == name)
    b = next(n for n in right.body if isinstance(n, ast.FunctionDef) and n.name == name)
    assert ast.dump(a) == ast.dump(b), name
for p in [patched, *stage.glob('tests/*.py')]: compile(p.read_text(), str(p), 'exec')
check = Path('/tmp/n081_codex/diffcheck')
(check/'scripts').mkdir(parents=True, exist_ok=True)
shutil.copyfile(source, check/'scripts/train_lf.py')
r = subprocess.run(['patch', '--batch', '-p1', '-d', str(check), '-i', str(stage/'train_lf.diff')], capture_output=True, text=True, check=True)
assert (check/'scripts/train_lf.py').read_bytes() == patched.read_bytes()
# Check documented shell snippets without executing the production commands.
readme = (stage/'README.md').read_text()
for i, block in enumerate(readme.split('```bash\n')[1:]):
    block = block.split('```', 1)[0]
    result = subprocess.run(['bash', '-n'], input=block, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
import torch, timm, numpy, pandas, pyarrow
n074 = repo/'.agent-research/results/N074-dense-privileged-supervision-auxiliary-d'
paths = [source, repo/'bsccm/encoders.py', repo/'scripts/pack_sbiad452.py',
         n074/'verify/test_dense_aux_integrated.py', n074/'codex/tests/test_dense_aux.py',
         *stage.glob('tests/*.py'), patched, stage/'train_lf.diff', stage/'README.md',
         stage/'tests/n074_regression.log', stage/'tests/ifc_cpu.log']
provenance = dict(
    baseline=str(source), baseline_sha256=sha(source), patched_sha256=sha(patched),
    python=sys.version, versions={m.__name__:m.__version__ for m in (torch,timm,numpy,pandas,pyarrow)},
    device='cpu', slurm_submitted=False, repository_modified=False,
    scratch='/tmp/n081_codex',
    tests={'n074_reused':{'count':7,'status':'passed','log':'tests/n074_regression.log'},
           'ifc':{'count':6,'status':'passed','log':'tests/ifc_cpu.log'},
           'two_step_protocols':['b1to0','b0to1'],
           'baseline_functions_ast_equal':['prep','fwd','evaluate'],
           'diff_reproduces_full_copy':True, 'readme_commands_bash_syntax':'passed'},
    n074_source_rows=json.loads(Path('/tmp/n081_codex/n074/real_rows.json').read_text()),
    synthetic_ifc={'n_cells':24,'donors':{'sample0':8,'sample1':8,'sample2':8},
                   'train':['sample0','sample1'],'test':'sample2'},
    files_sha256={str(p):sha(p) for p in paths})
(stage/'tests/provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
print(r.stdout.strip())
print('Full copy == applied diff; BSCCM prep/fwd/evaluate AST unchanged; Python and documented bash syntax valid.')
print('Baseline SHA256:',sha(source))
print('Patched SHA256:',sha(patched))
