"""Path setup for the Memory Study test suite.

Mirrors the repo-wide convention (each test package bootstraps src/ onto
sys.path itself). These tests are hermetic: no jax, no craftax, no network,
no real checkpoints.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))