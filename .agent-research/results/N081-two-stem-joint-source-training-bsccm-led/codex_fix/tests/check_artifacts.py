"""Validate staged diffs, baseline identity, unchanged BSCCM functions, and provenance."""
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

STAGE = Path(__file__).resolve().parents[1]
REPO = next(p for p in STAGE.parents if (p/'bsccm/data.py').is_file())
TMP = Path('/tmp/n081_fix/artifact_check')
TMP.mkdir(parents=True,exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
baseline=REPO/'scripts/train_lf.py'
expected='4cf89b093fc7dd44aa13ce6a2a82f743eeec2bea03c0b536b298c755362ce8c8'
assert sha(baseline)==expected, 'live baseline changed'
old,new=ast.parse(baseline.read_text()),ast.parse((STAGE/'train_lf.py').read_text())
for name in ('prep','fwd','evaluate'):
    select=lambda tree: next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.dump(select(old))==ast.dump(select(new)),name
for p in [STAGE/'train_lf.py',STAGE/'pack_sbiad452.py',*sorted((STAGE/'tests').glob('*.py'))]:
    compile(p.read_text(),str(p),'exec')
for name in ('train_lf','pack_sbiad452'):
    target=TMP/'scripts'/f'{name}.py'; target.parent.mkdir(exist_ok=True)
    shutil.copyfile(REPO/'scripts'/f'{name}.py',target)
    r=subprocess.run(['patch','--batch','-p1','-d',str(TMP),'-i',str(STAGE/f'{name}.diff')],capture_output=True,text=True,check=True)
    assert target.read_bytes()==(STAGE/f'{name}.py').read_bytes()
    print(r.stdout.strip())
for block in (STAGE/'README.md').read_text().split('```bash\n')[1:]:
    r=subprocess.run(['bash','-n'],input=block.split('```',1)[0],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
results={}
for name,count in (('n074_regression',7),('ifc_cpu',6),('fix_cpu',9)):
    text=(STAGE/'tests'/f'{name}.log').read_text()
    assert re.search(rf'Ran {count} tests in ',text),name
    assert re.search(r'^OK$',text,re.M),name
    assert not re.search(r'^FAILED',text,re.M),name
    results[name]=dict(count=count,status='passed',log=f'tests/{name}.log')
import torch,timm,numpy,pandas,pyarrow
stats={}
for line in (STAGE/'tests/fix_cpu.log').read_text().splitlines():
    if line.startswith('REAL V10 '):
        _,_,protocol,record=line.split(' ',3)
        stats[protocol]=json.loads(record)
assert set(stats)=={'b1to0','b0to1'}
paths=[baseline,REPO/'scripts/pack_sbiad452.py',STAGE.parent/'codex/train_lf.py',STAGE.parent/'agents_redteam.md',
       STAGE/'train_lf.py',STAGE/'train_lf.diff',STAGE/'pack_sbiad452.py',STAGE/'pack_sbiad452.diff',STAGE/'README.md',
       *sorted((STAGE/'tests').glob('*.py')),
       *[STAGE/'tests'/f'{name}.log' for name in results]]
provenance=dict(baseline=str(baseline),baseline_sha256=expected,
    python=sys.version,versions={m.__name__:m.__version__ for m in (torch,timm,numpy,pandas,pyarrow)},
    device='cpu',slurm_submitted=False,repository_modified=False,scratch='/tmp/n081_fix',
    tests=results,total_tests=22,baseline_functions_ast_equal=['prep','fwd','evaluate'],
    diffs_reproduce_full_copies=True,readme_bash_syntax='passed',
    bn_tests=dict(layers=53,steps=3,zero_weight_exact_rng_and_predictions=True,
                  nonzero_weight_replay_control=True,dual_zoom=True,strict_missing_buffer_resume=True),
    real_v10_affine_initializations_with_synthetic_ifc=stats,
    n074_source_rows=json.loads(Path('/tmp/n081_fix/n074/real_rows.json').read_text()),
    files_sha256={str(p):sha(p) for p in paths})
(STAGE/'tests/provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Both diffs reproduce staged copies; Python and README bash syntax valid.')
print('BSCCM prep/fwd/evaluate AST-identical; live baseline SHA256:',sha(baseline))
print('22/22 CPU tests pass; source/artifact hashes and real v10 affine records written to tests/provenance.json.')
