"""Reuse N074's integrated suite against today's baseline and the N081 full copy.

Only path redirection and the evolved baseline's _run_args() comparison change.
The original N074 assertion used vars(a), predating default-off dense/aux options.
"""
from pathlib import Path
import types

STAGE = Path(__file__).resolve().parents[1]
REPO = next(p for p in STAGE.parents if (p/'bsccm/data.py').is_file())
N074 = REPO/'.agent-research/results/N074-dense-privileged-supervision-auxiliary-d'
SOURCE = N074/'verify/test_dense_aux_integrated.py'

def load():
    source = SOURCE.read_text()
    source = source.replace("TMP = Path('/tmp/n074_verify/test')", "TMP = Path('/tmp/n081_codex/n074')")
    source = source.replace("left['a'].__dict__, right['_run_args']()", "left['_run_args'](), right['_run_args']()")
    source = source.replace("str(STAGE/'pack_fluor_aux.py')", "str(REPO/'scripts/pack_fluor_aux.py')")
    module = types.ModuleType('n074_reused')
    module.__file__ = str(SOURCE)
    exec(compile(source, str(SOURCE), 'exec'), module.__dict__)
    module.STAGE = STAGE
    module.ORIGINAL = REPO/'scripts/train_lf.py'
    module.PATCHED = STAGE/'train_lf.py'
    return module
