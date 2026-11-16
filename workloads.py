"""Deterministic finite calibration correction workloads."""
import random
from fractions import Fraction as F
from ledger import Observation


def workload(size, kind, batches=20):
    if size < 4 or size % 2 or kind not in ("stable", "split") or batches < 1:
        raise ValueError("even size >=4, stable/split, positive batch count required")
    nodes = tuple(f"N{i:03d}" for i in range(size))
    half = size // 2
    initial, interior = {}, []
    for start in (0, half):
        for i in range(start, start + half):
            for j in range(i + 1, start + half):
                key = f"edge:{i}:{j}"
                initial[key] = Observation(nodes[i], nodes[j], j - i, 1)
                interior.append(key)
    bridge = Observation(nodes[half - 1], nodes[half], 1, 1)
    initial["bridge"] = bridge
    rng = random.Random(20261116 + size)
    changes = []
    for step in range(batches):
        key = interior[rng.randrange(len(interior))]
        old = initial[key]
        batch = {key: Observation(old.u, old.v,
                                 old.d + F(rng.randrange(-4, 5), 3),
                                 F(rng.randrange(1, 5), 2))}
        if kind == "split":
            batch["bridge"] = None if step % 2 == 0 else Observation(
                bridge.u, bridge.v, F(step + 2, 3), F(step % 3 + 1))
        changes.append(batch)
    pairs = ((nodes[0], nodes[half - 1]),
             (nodes[half], nodes[-1]),
             (nodes[0], nodes[-1]),
             (nodes[-1], nodes[0]),
             (nodes[half - 1], nodes[half]),
             (nodes[0], nodes[0]))
    return nodes, initial, tuple(changes), pairs
