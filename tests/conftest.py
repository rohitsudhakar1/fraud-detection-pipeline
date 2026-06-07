"""Make project root importable for tests so `from src.X import Y` works
when pytest is invoked from the repo root.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
