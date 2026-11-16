"""Atomic in-memory ledger of independent difference observations."""
from dataclasses import dataclass
from fractions import Fraction
from collections.abc import Mapping


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
class Snapshot:
    revision: int
    nodes: tuple
    rows: tuple


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
        self._revision = 0

    @property
    def nodes(self):
        return self._nodes

    @property
    def revision(self):
        return self._revision

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
        self._rows = candidate
        self._revision += 1
        return self._revision
