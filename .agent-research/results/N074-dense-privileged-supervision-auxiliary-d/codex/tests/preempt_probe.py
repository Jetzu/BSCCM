"""Test-only delay allows deterministic SIGTERM during a 64-cell CLI smoke."""
import importlib.util
from pathlib import Path
import time

path = Path(__file__).resolve().parents[1]/'pack_fluor_aux.py'
spec = importlib.util.spec_from_file_location('n074_packer_probe', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = module.write_job

def slow_write(rows):
    time.sleep(.15)
    return original(rows)

module.write_job = slow_write
module.main()
