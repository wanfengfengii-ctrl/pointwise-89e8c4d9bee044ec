"""Exact rational LP/QP engine for the bounded-calibration extension.

All arithmetic uses fractions.Fraction; no floating point is involved.
The LP solver is a two-phase simplex with Bland's rule (finite by
construction); the QP solver is a finite enumeration of KKT active sets.
Sizes are tiny by contract (<=8 potentials, <=10 bound inequalities).
"""
from fractions import Fraction
from itertools import combinations


def _rref_solve(matrix, rhs, n_cols):
    """Solve matrix·x = rhs exactly over the rationals.

    Returns (particular, unique) where the particular solution sets free
    coordinates to zero and `unique` tells whether the solution is the only
    one; returns None when the system is inconsistent.
    """
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    pivots = []
    lead = 0
    for col in range(n_cols):
        pivot = next((r for r in range(lead, len(rows)) if rows[r][col]), None)
        if pivot is None:
            continue
        rows[lead], rows[pivot] = rows[pivot], rows[lead]
        p = rows[lead][col]
        rows[lead] = [v / p for v in rows[lead]]
        for r in range(len(rows)):
            if r != lead:
                scale = rows[r][col]
                if scale:
                    rows[r] = [v - scale * w for v, w in zip(rows[r], rows[lead])]
        pivots.append(col)
        lead += 1
        if lead == len(rows):
            break
    if any(not any(row[:n_cols]) and row[n_cols] for row in rows):
        return None
    particular = [Fraction(0)] * n_cols
    for r, col in enumerate(pivots):
        particular[col] = rows[r][n_cols]
    return particular, len(pivots) == n_cols


class _Simplex:
    """Two-phase exact simplex: maximize objective·x s.t. matrix·x = rhs, x >= 0."""

    def __init__(self, objective, matrix, rhs):
        self.n = len(objective)
        self.matrix = [list(row) for row in matrix]
        self.rhs = list(rhs)
        for i, value in enumerate(self.rhs):
            if value < 0:
                self.matrix[i] = [-v for v in self.matrix[i]]
                self.rhs[i] = -value
        self.m = len(self.matrix)
        self.artificials = self.m
        self.width = self.n + self.artificials
        self.tableau = [
            self.matrix[i] + [Fraction(1) if k == i else Fraction(0)
                              for k in range(self.artificials)]
            for i in range(self.m)
        ]
        self.basis = [self.n + i for i in range(self.m)]
        self.objective = list(objective) + [Fraction(0)] * self.artificials

    def _pivot(self, row, col):
        inverse = Fraction(1) / self.tableau[row][col]
        self.tableau[row] = [v * inverse for v in self.tableau[row]]
        self.rhs[row] = self.rhs[row] * inverse
        prow = self.tableau[row]
        for i in range(self.m):
            if i == row:
                continue
            factor = self.tableau[i][col]
            if factor:
                self.tableau[i] = [v - factor * p
                                   for v, p in zip(self.tableau[i], prow)]
                self.rhs[i] = self.rhs[i] - factor * self.rhs[row]
        self.basis[row] = col

    def _optimize(self, objective, allowed):
        # Bland's rule on both entering and leaving guarantees termination.
        while True:
            enter = -1
            for col in range(self.width):
                if not allowed[col]:
                    continue
                reduced = objective[col]
                for i in range(self.m):
                    coeff = objective[self.basis[i]]
                    if coeff:
                        reduced -= coeff * self.tableau[i][col]
                if reduced > 0:
                    enter = col
                    break
            if enter < 0:
                return "optimal"
            leave = -1
            best = None
            for i in range(self.m):
                if self.tableau[i][enter] > 0:
                    ratio = self.rhs[i] / self.tableau[i][enter]
                    if (best is None or ratio < best or
                            (ratio == best and self.basis[i] < self.basis[leave])):
                        best = ratio
                        leave = i
            if leave < 0:
                return "unbounded"
            self._pivot(leave, enter)

    def solve(self):
        """Returns (status, value, point); status is optimal/unbounded/infeasible."""
        # Phase I: maximize minus the sum of artificial variables.
        phase_one = [Fraction(0)] * self.n + [Fraction(-1)] * self.artificials
        self._optimize(phase_one, [True] * self.width)
        for i in range(self.m):
            if self.basis[i] >= self.n and self.rhs[i] != 0:
                return "infeasible", None, None
        # Drive zero-level basic artificials out; drop redundant rows.
        for i in range(self.m):
            if self.basis[i] >= self.n:
                col = next((j for j in range(self.n) if self.tableau[i][j]), None)
                if col is not None:
                    self._pivot(i, col)
        keep = [i for i in range(self.m) if self.basis[i] < self.n]
        if len(keep) < self.m:
            self.tableau = [self.tableau[i] for i in keep]
            self.rhs = [self.rhs[i] for i in keep]
            self.basis = [self.basis[i] for i in keep]
            self.m = len(keep)
        # Phase II: real objective, artificial columns banned from entering.
        allowed = [True] * self.n + [False] * self.artificials
        if self._optimize(self.objective, allowed) == "unbounded":
            return "unbounded", None, None
        point = [Fraction(0)] * self.n
        for i in range(self.m):
            point[self.basis[i]] = self.rhs[i]
        value = sum((c * x for c, x in zip(self.objective, point)), Fraction(0))
        return "optimal", value, point


def _standard_form(n_vars, equalities, inequalities, objective):
    """Convert a free-variable LP to max c·y s.t. M·y = k, y >= 0.

    Equalities and inequalities are (row, value) pairs meaning row·x = value
    and row·x <= value; every variable is free and gets split into a
    nonnegative pair. Slack columns are appended for the inequalities.
    """
    n_ineq = len(inequalities)
    matrix = []
    rhs = []
    for row, value in equalities:
        matrix.append(list(row) + [-v for v in row] + [Fraction(0)] * n_ineq)
        rhs.append(value)
    for k, (row, value) in enumerate(inequalities):
        matrix.append(list(row) + [-v for v in row] +
                      [Fraction(1) if i == k else Fraction(0)
                       for i in range(n_ineq)])
        rhs.append(value)
    obj = list(objective) + [-v for v in objective] + [Fraction(0)] * n_ineq
    return obj, matrix, rhs


def lp_feasible_point(n_vars, equalities, inequalities):
    """One point of {equalities, inequalities}, or None when infeasible."""
    obj, matrix, rhs = _standard_form(n_vars, equalities, inequalities,
                                      [Fraction(0)] * n_vars)
    status, _, point = _Simplex(obj, matrix, rhs).solve()
    if status != "optimal":
        return None
    return tuple(point[i] - point[i + n_vars] for i in range(n_vars))


def lp_extremum(n_vars, equalities, inequalities, objective, maximize=True):
    """Exact optimum of objective·x over the polyhedron; None when unbounded."""
    obj, matrix, rhs = _standard_form(n_vars, equalities, inequalities, objective)
    if not maximize:
        obj = [-v for v in obj]
    status, value, _ = _Simplex(obj, matrix, rhs).solve()
    if status == "infeasible":
        raise AssertionError("extremum requested over an empty polyhedron")
    if status == "unbounded":
        return None
    return value if maximize else -value


def qp_optimal_point(information, rhs, ineq_rows, ineq_vals):
    """One minimizer of x·H·x - 2·g·x subject to ineq_rows·x <= ineq_vals.

    `information` is the dense symmetric PSD matrix H = Σ w·b·bᵀ and `rhs`
    is g = Σ w·d·b of the least-squares objective. Enumerates KKT active
    sets in increasing size; by Carathéodory a feasible KKT point with at
    most min(m, n) active inequalities always exists when the QP is
    feasible, and every KKT point of a convex QP is a global minimizer.
    """
    n = len(information)
    m = len(ineq_rows)
    for size in range(min(m, n) + 1):
        for active in combinations(range(m), size):
            eq_rows = []
            eq_vals = []
            for i in range(n):
                eq_rows.append(list(information[i]) +
                               [ineq_rows[k][i] for k in active])
                eq_vals.append(rhs[i])
            for k in active:
                eq_rows.append(list(ineq_rows[k]) + [Fraction(0)] * size)
                eq_vals.append(ineq_vals[k])
            solved = _rref_solve(eq_rows, eq_vals, n + size)
            if solved is None:
                continue
            particular, unique = solved
            if unique:
                point = particular[:n]
                multipliers = particular[n:]
                if all(value >= 0 for value in multipliers) and all(
                        sum((a * x for a, x in zip(row, point)), Fraction(0)) <= value
                        for row, value in zip(ineq_rows, ineq_vals)):
                    return tuple(point)
            else:
                inequalities = [(list(ineq_rows[k]) + [Fraction(0)] * size,
                                 ineq_vals[k]) for k in range(m)]
                for idx in range(size):
                    inequalities.append(([Fraction(0)] * n +
                                          [Fraction(-1) if j == idx else Fraction(0)
                                           for j in range(size)], Fraction(0)))
                point = lp_feasible_point(n + size, list(zip(eq_rows, eq_vals)),
                                          inequalities)
                if point is not None:
                    return tuple(point[:n])
    raise AssertionError("no KKT point found: QP assumed feasible")
