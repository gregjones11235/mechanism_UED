"""Path setup for the E1 formal teacher test suite.

Mirrors the repo-wide convention (each test package bootstraps
``src/`` onto ``sys.path`` itself). These tests never call a real
external API and never train.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
# repo root: required ONLY for the sanctioned d052.achievements REGISTRY
# import in task_specs (pure stdlib); E1 runtime imports nothing else
# from d052.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
