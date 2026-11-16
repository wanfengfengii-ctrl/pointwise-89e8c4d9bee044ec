"""Independent full reference: last-node grounding and forward/back solve."""
from fractions import Fraction as F
from ledger import Ledger, name


def _solve_many(matrix, right):
    n = len(matrix)
    if not n:
        return []
    width = len(right[0])
    a = [list(row) + list(rhs) for row, rhs in zip(matrix, right)]
    for k in range(n):
        pivot = next(i for i in range(k, n) if a[i][k])
        a[k], a[pivot] = a[pivot], a[k]
        for i in range(k + 1, n):
            if not a[i][k]:
                continue
            scale = a[i][k] / a[k][k]
            a[i][k] = F(0)
            for j in range(k + 1, n + width):
                a[i][j] -= scale * a[k][j]
    out = [[F(0)] * width for _ in range(n)]
    for i in range(n - 1, -1, -1):
        for j in range(width):
            out[i][j] = (a[i][n + j] - sum(
                (a[i][k] * out[k][j] for k in range(i + 1, n)), F(0))) / a[i][i]
    return out


class Reference:
    def __init__(self, nodes):
        self.ledger = Ledger(nodes)
        self._revision = -1

    def _refresh(self):
        if self._revision == self.ledger.revision:
            return
        snap = self.ledger.snapshot()
        # Union-find deliberately differs from the production component search.
        parent = {n: n for n in snap.nodes}

        def root(n):
            while parent[n] != n:
                n = parent[n]
            return n

        for _, obs in snap.rows:
            x, y = root(obs.u), root(obs.v)
            parent[x] = y
        groups = {}
        for n in snap.nodes:
            groups.setdefault(root(n), []).append(n)
        membership, fits = {}, []
        for group in groups.values():
            group = tuple(sorted(group))
            index = len(fits)
            for n in group:
                membership[n] = index
            free = {n: i for i, n in enumerate(group[:-1])}
            count = len(free)
            matrix = [[F(0) for _ in free] for _ in free]
            rhs = [F(0) for _ in free]
            for _, obs in snap.rows:
                if root(obs.u) != root(group[0]):
                    continue
                u, v = free.get(obs.u), free.get(obs.v)
                if u is not None:
                    matrix[u][u] += obs.w
                    rhs[u] -= obs.w * obs.d
                if v is not None:
                    matrix[v][v] += obs.w
                    rhs[v] += obs.w * obs.d
                if u is not None and v is not None:
                    matrix[u][v] -= obs.w
                    matrix[v][u] -= obs.w
            solved = _solve_many(matrix, [
                [rhs[i]] + [F(i == j) for j in range(count)] for i in range(count)])
            values = tuple(row[0] for row in solved) + (F(0),)
            cov = tuple(tuple(row[1:]) + (F(0),) for row in solved)
            cov += (tuple(F(0) for _ in group),)
            fits.append(({n: i for i, n in enumerate(group)}, values, cov))
        self._membership, self._fits = membership, fits
        self._revision = snap.revision

    def query(self, a, b):
        name(a)
        name(b)
        self._refresh()
        if a not in self._membership or b not in self._membership:
            raise ValueError("unknown node")
        if self._membership[a] != self._membership[b]:
            return None
        positions, values, cov = self._fits[self._membership[a]]
        i, j = positions[a], positions[b]
        return values[j] - values[i], cov[i][i] + cov[j][j] - cov[i][j] - cov[j][i]
