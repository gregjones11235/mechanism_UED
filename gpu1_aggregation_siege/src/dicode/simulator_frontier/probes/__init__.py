"""Read-only compatibility probes (python -m entry points only).

Modules here are NEVER imported by ``simulator_frontier.__init__``: they are
opt-in diagnostic drivers and must stay side-effect free at import time.
Zero parameter updates anywhere in this package.
"""
