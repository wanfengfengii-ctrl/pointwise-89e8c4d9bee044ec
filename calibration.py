"""Full recalibration after each accepted ledger batch; exact arithmetic."""
from dataclasses import dataclass
from fractions import Fraction as F
from types import MappingProxyType


@dataclass(frozen=True)
class Fit:
    nodes: tuple
    positions: object
    values: tuple
    covariance: tuple


@dataclass(frozen=True)
class State:
    revision: int
    membership: object
    fits: tuple


def _inverse(matrix):
    n = len(matrix)
    rows = [list(row) + [F(i == j) for j in range(n)]
            for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = next(i for i in range(col, n) if rows[i][col])
        rows[col], rows[pivot] = rows[pivot], rows[col]
        p = rows[col][col]
        rows[col] = [x / p for x in rows[col]]
        for i in range(n):
            if i == col or not rows[i][col]:
                continue
            scale = rows[i][col]
            rows[i] = [x - scale * y for x, y in zip(rows[i], rows[col])]
    return [row[n:] for row in rows]


def _fit(nodes, observations):
    # The first node is grounded. All returned differences are gauge invariant.
    size = len(nodes)
    positions = {node: i for i, node in enumerate(nodes)}
    matrix = [[F(0) for _ in range(size - 1)] for _ in range(size - 1)]
    rhs = [F(0) for _ in range(size - 1)]
    for obs in observations:
        terms = [(positions[obs.u] - 1, -1), (positions[obs.v] - 1, 1)]
        terms = [(i, sign) for i, sign in terms if i >= 0]
        for i, sign in terms:
            rhs[i] += sign * obs.w * obs.d
            for j, other in terms:
                matrix[i][j] += sign * other * obs.w
    inverse = _inverse(matrix)
    values = (F(0),) + tuple(sum((x * y for x, y in zip(row, rhs)), F(0))
                             for row in inverse)
    covariance = (tuple(F(0) for _ in nodes),) + tuple(
        (F(0),) + tuple(row) for row in inverse)
    return Fit(nodes, MappingProxyType(positions), values, covariance)


class Calibrator:
    def __init__(self, ledger):
        self.ledger = ledger
        self._state = None

    def current(self):
        if self._state is not None and self._state.revision == self.ledger.revision:
            return self._state
        snapshot = self.ledger.snapshot()
        adjacency = {node: set() for node in snapshot.nodes}
        for _, obs in snapshot.rows:
            adjacency[obs.u].add(obs.v)
            adjacency[obs.v].add(obs.u)
        membership = {}
        groups = []
        for node in snapshot.nodes:
            if node in membership:
                continue
            index = len(groups)
            pending = [node]
            members = []
            membership[node] = index
            while pending:
                here = pending.pop()
                members.append(here)
                for neighbor in sorted(adjacency[here]):
                    if neighbor not in membership:
                        membership[neighbor] = index
                        pending.append(neighbor)
            groups.append(tuple(sorted(members)))
        observations = [[] for _ in groups]
        for _, obs in snapshot.rows:
            observations[membership[obs.u]].append(obs)
        fits = tuple(_fit(group, rows) for group, rows in zip(groups, observations))
        state = State(snapshot.revision, MappingProxyType(membership), fits)
        self._state = state
        return state
