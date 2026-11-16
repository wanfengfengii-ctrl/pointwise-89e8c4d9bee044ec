"""Atomic in-memory ledger of independent difference observations."""
from dataclasses import dataclass
from fractions import Fraction
from collections.abc import Mapping
from types import MappingProxyType

# Size envelope of the bounded-calibration extension (Bound/apply_bounds/
# Analyzer.range). The original observation interfaces are not limited.
BOUNDED_MAX_NODES = 8
BOUNDED_MAX_OBSERVATIONS = 40
BOUNDED_MAX_ENDPOINTS = 10


def rational(value):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise ValueError("expected int, rational string or Fraction")
    try:
        return Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError("invalid rational") from exc


def name(value):
    if not isinstance(value, str) or not value:
        raise ValueError("expected nonempty string")
    return value


@dataclass(frozen=True)
class Observation:
    u: str
    v: str
    d: Fraction
    w: Fraction

    def __post_init__(self):
        name(self.u)
        name(self.v)
        if self.u == self.v:
            raise ValueError("self observation")
        object.__setattr__(self, "d", rational(self.d))
        object.__setattr__(self, "w", rational(self.w))
        if self.w <= 0:
            raise ValueError("weight must be positive")


@dataclass(frozen=True)
class Bound:
    """Physical bound lower <= x_v - x_u <= upper; None means unbounded."""
    u: str
    v: str
    lower: object = None
    upper: object = None

    def __post_init__(self):
        name(self.u)
        name(self.v)
        if self.u == self.v:
            raise ValueError("self bound")
        lower = None if self.lower is None else rational(self.lower)
        upper = None if self.upper is None else rational(self.upper)
        if lower is None and upper is None:
            raise ValueError("bound needs at least one finite endpoint")
        if lower is not None and upper is not None and lower > upper:
            raise ValueError("lower endpoint exceeds upper endpoint")
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class Snapshot:
    revision: int
    nodes: tuple
    rows: tuple


def _bounds_feasible(nodes, bounds):
    """Exact negative-cycle test over the difference constraints of `bounds`.

    Each finite endpoint contributes an edge x_v <= x_u + c; the system is
    feasible iff no directed cycle has negative total weight (Bellman-Ford
    from a virtual super-source, exact Fraction arithmetic).
    """
    index = {node: i for i, node in enumerate(nodes)}
    edges = []
    for bound in bounds:
        u, v = index[bound.u], index[bound.v]
        if bound.upper is not None:
            edges.append((u, v, bound.upper))
        if bound.lower is not None:
            edges.append((v, u, -bound.lower))
    if not nodes:
        return True
    dist = [Fraction(0)] * len(nodes)
    for _ in range(len(nodes)):
        updated = False
        for u, v, weight in edges:
            if dist[v] > dist[u] + weight:
                dist[v] = dist[u] + weight
                updated = True
        if not updated:
            return True
    return False


class Ledger:
    def __init__(self, nodes):
        if isinstance(nodes, (str, bytes)):
            raise ValueError("nodes must be an iterable of names")
        try:
            ordered = tuple(name(n) for n in nodes)
        except TypeError as exc:
            raise ValueError("nodes must be iterable") from exc
        if len(set(ordered)) != len(ordered):
            raise ValueError("duplicate node")
        self._nodes = tuple(sorted(ordered))
        self._known = frozenset(ordered)
        self._rows = {}
        self._bounds = {}
        self._revision = 0

    @property
    def nodes(self):
        return self._nodes

    @property
    def revision(self):
        return self._revision

    @property
    def bounds(self):
        """Read-only view of the current bounds, keyed by bound ID."""
        return MappingProxyType(self._bounds)

    def snapshot(self):
        return Snapshot(self._revision, self._nodes,
                        tuple(sorted(self._rows.items())))

    def apply(self, batch):
        """Upsert Observation values; None deletes an existing ID, atomically."""
        if not isinstance(batch, Mapping):
            raise ValueError("batch must be a mapping")
        changes = []
        for key, value in batch.items():
            name(key)
            if value is None:
                if key not in self._rows:
                    raise ValueError("delete of unknown observation")
            elif type(value) is Observation:
                if value.u not in self._known or value.v not in self._known:
                    raise ValueError("unknown endpoint")
            else:
                raise ValueError("expected Observation or None")
            changes.append((key, value))
        candidate = self._rows.copy()
        for key, value in changes:
            if value is None:
                del candidate[key]
            else:
                candidate[key] = value
        if self._bounds and len(candidate) > BOUNDED_MAX_OBSERVATIONS:
            raise ValueError("too many observations for a bounded ledger")
        self._rows = candidate
        self._revision += 1
        return self._revision

    def apply_bounds(self, batch):
        """Upsert Bound values; None deletes an existing bound ID, atomically.

        Bound IDs live in their own namespace, independent of observation
        IDs. The whole candidate batch must be jointly feasible and within
        the bounded-interface size limits, otherwise nothing changes.
        """
        if not isinstance(batch, Mapping):
            raise ValueError("batch must be a mapping")
        changes = []
        for key, value in batch.items():
            name(key)
            if value is None:
                if key not in self._bounds:
                    raise ValueError("delete of unknown bound")
            elif type(value) is Bound:
                if value.u not in self._known or value.v not in self._known:
                    raise ValueError("unknown endpoint")
            else:
                raise ValueError("expected Bound or None")
            changes.append((key, value))
        candidate = self._bounds.copy()
        for key, value in changes:
            if value is None:
                del candidate[key]
            else:
                candidate[key] = value
        if len(self._nodes) > BOUNDED_MAX_NODES:
            raise ValueError("too many nodes for bounded calibration")
        if len(self._rows) > BOUNDED_MAX_OBSERVATIONS:
            raise ValueError("too many observations for bounded calibration")
        endpoints = sum((bound.lower is not None) + (bound.upper is not None)
                        for bound in candidate.values())
        if endpoints > BOUNDED_MAX_ENDPOINTS:
            raise ValueError("too many finite bound endpoints")
        if not _bounds_feasible(self._nodes, candidate.values()):
            raise ValueError("jointly infeasible bound batch")
        self._bounds = candidate
        self._revision += 1
        return self._revision
