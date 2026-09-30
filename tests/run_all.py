"""Run the repository's unittest suite with a process exit code for notebooks/CI."""
import sys
import unittest
from pathlib import Path


# Executing this file directly sets sys.path[0] to tests/, not the repository
# root.  Add the root explicitly so imports such as ``lib.aquant`` behave the
# same in local runs, CI, and the generated Kaggle notebook.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

suite = unittest.defaultTestLoader.discover(str(Path(__file__).resolve().parent))
result = unittest.TextTestRunner(verbosity=2).run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
