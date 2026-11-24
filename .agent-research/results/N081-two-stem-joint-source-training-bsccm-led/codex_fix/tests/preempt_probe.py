"""N074's SIGTERM packer test, redirected to the current read-only packer."""
import importlib.util
from pathlib import Path
import time
import sys
REPO = next(p for p in Path(__file__).resolve().parents if (p/'bsccm/data.py').is_file())
sys.path.insert(0, str(REPO))
path = REPO/'scripts/pack_fluor_aux.py'
spec = importlib.util.spec_from_file_location('n074_packer_probe', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
original = module.write_job

def slow_write(rows):
    time.sleep(.15)
    return original(rows)

module.write_job = slow_write
module.main()
