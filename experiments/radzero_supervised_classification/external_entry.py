"""Resolve the published namespace package before invoking the frozen evaluator.

RadZero/common has no __init__.py. Python would otherwise resolve our adjacent
common.py instead of that namespace, breaking imports in the published loader.
No model, preprocessing, checkpoint or evaluation code is changed here.
"""
import importlib.machinery
from pathlib import Path
import sys
import types

ROOT=Path(__file__).resolve().parents[2]
# The script directory also contains external.py. Keep it out of top-level
# import resolution; all our modules use explicit experiments.* imports.
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]
sys.path.insert(0,str(ROOT))
for name in ('common','external'):
    package=types.ModuleType(name)
    package.__path__=[str(ROOT/'RadZero'/name)]
    package.__spec__=importlib.machinery.ModuleSpec(name,loader=None,is_package=True)
    package.__spec__.submodule_search_locations=package.__path__
    sys.modules[name]=package

from experiments.radzero_supervised_classification.external import main

if __name__=='__main__':main()
