import os
from pathlib import Path
import sys

# Keep all generated packs, caches and test outputs outside the deliverable tree.
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/n080_codex/cache')
os.environ.setdefault('TORCH_HOME', '/tmp/n080_codex/torch')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
