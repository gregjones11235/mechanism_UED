"""Scoped pytest bootstrap for the d052 package.

Puts the gpu1_aggregation_siege package root on sys.path so `import d052`
resolves, without altering legacy gpu1 test collection (no root-level conftest).
"""
import os
import sys

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)
