#!/usr/bin/env python3
"""Run the staged harness while reusing the read-only repository's bsccm package."""
import runpy
import sys
from pathlib import Path

STAGE = Path(__file__).resolve().parents[1]
REPO = next(p for p in STAGE.parents if (p/'bsccm/data.py').is_file())
sys.path.insert(0, str(REPO))
import bsccm
bsccm.__path__.insert(0, str(STAGE/'bsccm'))
if __name__ == '__main__':
    sys.argv[0] = str(STAGE/'train_lf.py')
    runpy.run_path(sys.argv[0], run_name='__main__')
