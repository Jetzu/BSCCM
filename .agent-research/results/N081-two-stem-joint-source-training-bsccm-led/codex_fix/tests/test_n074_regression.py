"""Run all seven N074 integrated tests with the current baseline; CPU only."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from n074_support import load
import unittest

if __name__ == '__main__':
    module = load()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.DenseTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
