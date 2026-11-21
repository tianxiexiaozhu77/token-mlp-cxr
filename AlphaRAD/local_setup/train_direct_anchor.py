"""Run the shared audited classification comparison in AlphaRAD's environment."""
from pathlib import Path
import runpy
import sys

entry = Path(__file__).resolve().parents[2] / "experiments/direct_anchor_classification/train.py"
sys.path.insert(0, str(entry.parent))
runpy.run_path(str(entry), run_name="__main__")
