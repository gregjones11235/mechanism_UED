"""Deterministic state sampler: weights + seed -> entry selection (reproducible)."""
from __future__ import annotations

import hashlib
from typing import List, Sequence


def sample_entries(entries: Sequence, count: int, *, seed: int) -> List:
    if not entries:
        return []
    key = hashlib.sha256(f"{seed}".encode()).digest()
    order = []
    for i, entry in enumerate(entries):
        h = hashlib.sha256(key + getattr(entry, "state_id", str(i)).encode())
        order.append((h.hexdigest(), i))
    order.sort()
    picked = [entries[i] for _h, i in order]
    out = []
    for j in range(count):
        out.append(picked[j % len(picked)])
    return out