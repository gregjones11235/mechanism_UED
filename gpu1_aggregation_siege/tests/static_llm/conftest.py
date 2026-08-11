"""Path setup for the static-LLM-UED V1 test suite.

Mirrors the repo-wide convention (each test bootstraps ``src/`` onto
``sys.path`` itself) but centralized for the ``tests/static_llm/`` package.
These tests never call a real external API.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
