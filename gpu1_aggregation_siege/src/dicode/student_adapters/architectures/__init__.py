"""Vendored READ-ONLY architecture subsets (provenance-bound).

Each module here is a byte-faithful vendor of an audited upstream source file
(one recorded import fix for ``rmt16_anchor.py``; see ``rmt16_provenance``).
These modules import jax/flax at import time; they are therefore NEVER
imported by ``dicode.student_adapters.__init__`` (that package stays
jax-free).  Import them only from opt-in adapter modules such as
``dicode.student_adapters.rmt16_adapter``.

Training-side use (window forward for loss, optimizer, save) is OUT OF
SCOPE for this round: only the eval/forward subset is mounted read-only.
"""
