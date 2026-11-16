"""Independent correctness verification for the bounded-calibration extension.

Why the calibrator -> analyzer handoff covers ALL global optima
---------------------------------------------------------------
The joint objective q(x) = Σ w·(x_v − x_u − d)² = xᵀHx − 2gᵀx + const is
convex (H = Σ w·b·bᵀ is positive semidefinite) and bounded below by 0, so
over the nonempty polyhedron P = {x : A·x ≤ b} of physical bounds its
minimum q* is attained (Frank–Wolfe theorem). For any two minimizers x₁,
x₂ the midpoint is also a minimizer, and convexity forces
(x₁ − x₂)ᵀH(x₁ − x₂) = 0, hence H·x₁ = H·x₂: the gradient 2(Hx − g) is
constant over the optimal set. Moreover, for any x ∈ P, with d = x − x*,

    q(x) − q* = dᵀHd + 2(Hx* − g)ᵀd,

where the first term is ≥ 0 (H is PSD) and the second is ≥ 0 (first-order
optimality of x* over P). The sum vanishes iff both terms vanish, and
dᵀHd = 0 iff Hd = 0. Therefore

    S = argmin q = {x ∈ P : H·x = H·x*,  (Hx* − g)·x = (Hx* − g)·x*},

a polyhedron described by LINEAR constraints only. Calibrator.optimum()
ships exactly this description (bound inequalities, the normal-equation
rows H·x = H·x*, and the scalar optimality cut), computed from one
arbitrary minimizer x*; the description is provably independent of which
minimizer was found, so no optimal point is lost and no coordinate is ever
anchored. Analyzer.range(a, b) then solves two exact LPs projecting
x_b − x_a over all of S, including interior points of optimal faces.

Independence of this reference
------------------------------
This module shares no solver code with the production engine
(bounded.py). The production minimizer x* is re-verified here from first
principles: feasibility by direct substitution, and global optimality by
showing — via Fourier–Motzkin elimination, a different algorithm from the
production simplex — that no feasible descent direction exists (necessary
AND sufficient for a convex objective). The projection of x_b − x_a over
S is then recomputed by the same independent elimination. Hand-computed
expectations anchor the semantics; the randomized differential tests
anchor the computation. Checking points returned by the production solver
against itself would not constitute an independent reference, so nothing
here calls bounded.py.
"""
import random
import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction as F

from ledger import (Ledger, Observation, Bound, BOUNDED_MAX_NODES,
                    BOUNDED_MAX_OBSERVATIONS, BOUNDED_MAX_ENDPOINTS)
from calibration import Calibrator
from analysis import Analyzer


# ---------------------------------------------------------------------------
# Independent exact oracle: Fourier-Motzkin elimination over the rationals.
# ---------------------------------------------------------------------------

def _fm_eliminate(constraints, count):
    """Eliminate the first `count` variables from coeffs·z <= rhs systems."""
    cons = [(list(coeffs), rhs) for coeffs, rhs in constraints]
    for k in range(count):
        pos, neg, zero = [], [], []
        for coeffs, rhs in cons:
            if coeffs[k] > 0:
                pos.append((coeffs, rhs))
            elif coeffs[k] < 0:
                neg.append((coeffs, rhs))
            else:
                zero.append((coeffs, rhs))
        cons = zero
        for cp, rp in pos:
            for cn, rn in neg:
                p, q = cp[k], -cn[k]
                cons.append(([q * a + p * b for a, b in zip(cp, cn)],
                             q * rp + p * rn))
        # Exact deduplication keeps the exact elimination affordable.
        cons = list(dict.fromkeys((tuple(coeffs), rhs) for coeffs, rhs in cons))
        cons = [(list(coeffs), rhs) for coeffs, rhs in cons]
    return cons


def fm_feasible(n_vars, inequalities):
    """True iff the system row·x <= value (all variables free) is feasible."""
    cons = _fm_eliminate([(list(row), value) for row, value in inequalities],
                         n_vars)
    return all(rhs >= 0 for _, rhs in cons)


def fm_range(n_vars, equalities, inequalities, objective):
    """Independent (lo, hi) of objective·x over the polyhedron; None unbounded."""
    cons = []
    for row, value in equalities:
        cons.append((list(row) + [F(0)], value))
        cons.append(([-v for v in row] + [F(0)], -value))
    for row, value in inequalities:
        cons.append((list(row) + [F(0)], value))
    # t = objective·x, then eliminate every x and read off the bounds on t.
    cons.append(([-v for v in objective] + [F(1)], F(0)))
    cons.append((list(objective) + [F(-1)], F(0)))
    cons = _fm_eliminate(cons, n_vars)
    lo = hi = None
    for coeffs, rhs in cons:
        a = coeffs[n_vars]
        if a > 0:
            hi = rhs / a if hi is None else min(hi, rhs / a)
        elif a < 0:
            lo = rhs / a if lo is None else max(lo, rhs / a)
        elif rhs < 0:
            raise AssertionError("oracle: infeasible optimal set")
    return lo, hi


def _assemble(nodes, observations, bounds):
    """Independent assembly of H, g and the bound inequalities."""
    index = {node: i for i, node in enumerate(nodes)}
    n = len(nodes)
    information = [[F(0)] * n for _ in range(n)]
    rhs = [F(0)] * n
    for obs in observations:
        u, v = index[obs.u], index[obs.v]
        information[u][u] += obs.w
        information[v][v] += obs.w
        information[u][v] -= obs.w
        information[v][u] -= obs.w
        rhs[u] -= obs.w * obs.d
        rhs[v] += obs.w * obs.d
    rows, vals = [], []
    for bound in bounds:
        u, v = index[bound.u], index[bound.v]
        if bound.upper is not None:
            row = [F(0)] * n
            row[u] = F(-1)
            row[v] = F(1)
            rows.append(row)
            vals.append(bound.upper)
        if bound.lower is not None:
            row = [F(0)] * n
            row[u] = F(1)
            row[v] = F(-1)
            rows.append(row)
            vals.append(-bound.lower)
    return information, rhs, rows, vals


def oracle_range(nodes, observations, bounds, a, b, optimum):
    """Independent range of x_b - x_a over the optimal set.

    First re-verifies the production minimizer `optimum.point` from first
    principles, then projects with Fourier-Motzkin elimination.
    """
    n = len(nodes)
    information, rhs, rows, vals = _assemble(nodes, observations, bounds)
    point = optimum.point
    assert len(point) == n
    # 1. the production point must satisfy every physical bound.
    for row, val in zip(rows, vals):
        assert sum((c * x for c, x in zip(row, point)), F(0)) <= val, \
            "production minimizer violates a physical bound"
    # 2. global optimality: no feasible descent direction may exist.
    image = [sum((h * x for h, x in zip(row, point)), F(0))
             for row in information]
    gradient = [image[i] - rhs[i] for i in range(n)]
    tight = [(row, val) for row, val in zip(rows, vals)
             if sum((c * x for c, x in zip(row, point)), F(0)) == val]
    descent = [(list(row), F(0)) for row, _ in tight]
    descent.append((gradient, F(-1)))
    assert not fm_feasible(n, descent), \
        "production minimizer admits a feasible descent direction"
    # 3. independent projection over the optimal set S.
    equalities = [(row, image[i]) for i, row in enumerate(information)]
    equalities.append((gradient,
                       sum((g * x for g, x in zip(gradient, point)), F(0))))
    index = {node: i for i, node in enumerate(nodes)}
    objective = [F(0)] * n
    objective[index[a]] = F(-1)
    objective[index[b]] = F(1)
    return fm_range(n, equalities, list(zip(rows, vals)), objective)


def difference_constraints(nodes, bounds):
    """The bound endpoints as row·x <= value inequalities (for FM checks)."""
    _, _, rows, vals = _assemble(nodes, [], bounds)
    return list(zip(rows, vals))


def system(nodes):
    ledger = Ledger(nodes)
    calibrator = Calibrator(ledger)
    return ledger, calibrator, Analyzer(calibrator)


# ---------------------------------------------------------------------------
# Hand-computed cases: expectations derived on paper, not by any solver.
# ---------------------------------------------------------------------------

class HandComputed(unittest.TestCase):
    def test_coupled_optimum_splits_evenly(self):
        # AB=0, BC=0 (w=1); x_C - x_A >= 1. The optimum splits the unit gap
        # evenly: x_B - x_A = x_C - x_B = 1/2, and the optimal set is a line.
        ledger, _, analyzer = system(list("ABC"))
        ledger.apply({"ab": Observation("A", "B", 0, 1),
                      "bc": Observation("B", "C", 0, 1)})
        ledger.apply_bounds({"gap": Bound("A", "C", 1, None)})
        self.assertEqual(analyzer.range("A", "B"), (F(1, 2), F(1, 2)))
        self.assertEqual(analyzer.range("B", "C"), (F(1, 2), F(1, 2)))
        self.assertEqual(analyzer.range("A", "C"), (F(1), F(1)))
        self.assertEqual(analyzer.range("C", "A"), (F(-1), F(-1)))

    def test_redundant_bounds_collapse_to_strongest(self):
        ledger, _, analyzer = system(list("AB"))
        ledger.apply_bounds({"wide": Bound("A", "B", 0, 5),
                             "narrow": Bound("A", "B", 1, 2)})
        self.assertEqual(analyzer.range("A", "B"), (F(1), F(2)))
        self.assertEqual(analyzer.range("B", "A"), (F(-2), F(-1)))

    def test_equality_bound(self):
        ledger, _, analyzer = system(list("AB"))
        ledger.apply_bounds({"fixed": Bound("A", "B", 3, 3)})
        self.assertEqual(analyzer.range("A", "B"), (F(3), F(3)))
        self.assertEqual(analyzer.range("B", "A"), (F(-3), F(-3)))

    def test_no_observations_and_one_sided_unbounded(self):
        ledger, _, analyzer = system(list("ABC"))
        ledger.apply_bounds({"cap": Bound("A", "B", None, 2)})
        self.assertEqual(analyzer.range("A", "B"), (None, F(2)))
        self.assertEqual(analyzer.range("B", "A"), (F(-2), None))
        # C is completely free: doubly unbounded in both directions.
        self.assertEqual(analyzer.range("A", "C"), (None, None))
        self.assertEqual(analyzer.range("C", "A"), (None, None))

    def test_degenerate_isolated_node_and_no_bounds(self):
        ledger, _, analyzer = system(list("ABD"))
        ledger.apply({"ab": Observation("A", "B", 2, 1)})
        self.assertEqual(analyzer.range("A", "B"), (F(2), F(2)))
        self.assertEqual(analyzer.range("A", "D"), (None, None))
        self.assertEqual(analyzer.range("D", "A"), (None, None))
        self.assertEqual(analyzer.range("D", "D"), (F(0), F(0)))

    def test_unobserved_component_bounded_only(self):
        # AB observed, C free but chained: 0 <= x_C - x_B <= 3.
        ledger, _, analyzer = system(list("ABC"))
        ledger.apply({"ab": Observation("A", "B", 2, 1)})
        ledger.apply_bounds({"bc": Bound("B", "C", 0, 3)})
        self.assertEqual(analyzer.range("A", "C"), (F(2), F(5)))
        self.assertEqual(analyzer.range("C", "A"), (F(-5), F(-2)))
        self.assertEqual(analyzer.range("B", "C"), (F(0), F(3)))
        self.assertEqual(analyzer.range("A", "B"), (F(2), F(2)))

    def test_optimal_face_interior_counts(self):
        # Objective forces x_B - x_A = 2 only; bounds pin x_C into [0, 3]
        # relative to x_B. C = B is one optimum, but the range must keep the
        # whole face: the range is not collapsed to a single point.
        ledger, _, analyzer = system(list("ABC"))
        ledger.apply({"ab": Observation("A", "B", 2, 1)})
        ledger.apply_bounds({"bc": Bound("B", "C", 0, 3)})
        lo, hi = analyzer.range("A", "C")
        self.assertEqual((lo, hi), (F(2), F(5)))
        self.assertNotEqual(lo, hi)


# ---------------------------------------------------------------------------
# Contract: Bound construction, apply_bounds validation/atomicity, limits,
# revisions, immutability, and the observation/bound revision linkage.
# ---------------------------------------------------------------------------

class Contract(unittest.TestCase):
    def test_bound_construction_validation(self):
        for args in (("A", "A", 0, 1), ("", "B", 0, 1), ("A", "B", None, None),
                     ("A", "B", 2, 1), ("A", "B", True, 1), ("A", "B", 1.5, 2),
                     ("A", "B", "1/0", 2), ("A", "B", 1, False)):
            with self.assertRaises(ValueError, msg=args):
                Bound(*args)
        bound = Bound("A", "B", "1/2", F(3, 2))
        self.assertEqual((bound.lower, bound.upper), (F(1, 2), F(3, 2)))
        self.assertEqual(Bound("A", "B", None, 0).upper, F(0))
        self.assertIsNone(Bound("A", "B", None, 0).lower)
        with self.assertRaises(FrozenInstanceError):
            bound.upper = F(9)

    def test_apply_bounds_validation_and_atomicity(self):
        ledger, _, analyzer = system(list("ABC"))
        ledger.apply({"ab": Observation("A", "B", 2, 1)})
        ledger.apply_bounds({"bc": Bound("B", "C", 0, 3)})
        before = ledger.snapshot()
        revision = ledger.revision
        for batch in (
            {"bc": Bound("B", "C", 0, 2), "missing": None},       # unknown delete
            {"bc": None, "bad": Bound("B", "Z", 0, 1)},           # unknown endpoint
            {"bc": Bound("B", "C", 0, 2), "bad": ("B", "C", 0, 1)},  # bad type
            {"": Bound("B", "C", 0, 1)},                          # empty ID
            [("bc", None)],                                       # not a mapping
            {"bc": Bound("B", "C", 0, 3), "no": Bound("B", "C", None, -1)},
        ):
            with self.assertRaises(ValueError, msg=batch):
                ledger.apply_bounds(batch)
            self.assertEqual(ledger.revision, revision)
            self.assertEqual(ledger.snapshot(), before)
            self.assertEqual(analyzer.range("A", "C"), (F(2), F(5)))

    def test_revision_sharing_and_replay(self):
        ledger, _, _ = system(list("AB"))
        self.assertEqual(ledger.apply({"ab": Observation("A", "B", 1, 1)}), 1)
        self.assertEqual(ledger.apply_bounds({"b": Bound("A", "B", 0, 2)}), 2)
        self.assertEqual(ledger.apply({}), 3)
        self.assertEqual(ledger.apply_bounds({}), 4)
        self.assertEqual(ledger.apply_bounds({"b": Bound("A", "B", 0, 2)}), 5)
        self.assertEqual(ledger.apply_bounds({"b": None}), 6)
        self.assertEqual(ledger.revision, 6)

    def test_independent_id_namespaces(self):
        ledger, _, _ = system(list("ABC"))
        ledger.apply({"x": Observation("A", "B", 1, 1)})
        ledger.apply_bounds({"x": Bound("B", "C", 0, 1)})
        self.assertIn("x", ledger.bounds)
        self.assertEqual(dict(ledger.snapshot().rows)["x"].d, F(1))
        ledger.apply_bounds({"x": None})
        self.assertEqual(dict(ledger.snapshot().rows)["x"].d, F(1))

    def test_size_limits(self):
        # More than 8 nodes: apply_bounds and range reject.
        wide, _, wide_analyzer = system([f"N{i}" for i in range(9)])
        with self.assertRaises(ValueError):
            wide.apply_bounds({})
        with self.assertRaises(ValueError):
            wide_analyzer.range("N0", "N1")
        # More than 40 observations: apply keeps its original capacity, but
        # apply_bounds and range reject the ledger.
        busy, _, busy_analyzer = system(["A", "B"])
        busy.apply({f"o{i}": Observation("A", "B", i, 1) for i in range(41)})
        self.assertEqual(len(busy.snapshot().rows), 41)
        with self.assertRaises(ValueError):
            busy.apply_bounds({})
        with self.assertRaises(ValueError):
            busy_analyzer.range("A", "B")
        # With bounds present, apply must reject candidate observations > 40.
        ledger, _, _ = system(["A", "B"])
        ledger.apply({f"o{i}": Observation("A", "B", i, 1) for i in range(40)})
        ledger.apply_bounds({"b": Bound("A", "B", 0, 1)})
        revision = ledger.revision
        with self.assertRaises(ValueError):
            ledger.apply({"extra": Observation("A", "B", 0, 1)})
        self.assertEqual(ledger.revision, revision)
        ledger.apply({"o0": None})  # deletions still work
        self.assertEqual(len(ledger.snapshot().rows), 39)
        # More than 10 finite endpoints (a two-sided bound counts twice,
        # even when both endpoints are equal).
        capped, _, _ = system(list("ABCDEF"))
        capped.apply_bounds({f"b{i}": Bound("A", "B", 0, 1) for i in range(5)})
        self.assertEqual(sum((b.lower is not None) + (b.upper is not None)
                             for b in capped.bounds.values()),
                         BOUNDED_MAX_ENDPOINTS)
        with self.assertRaises(ValueError):
            capped.apply_bounds({"one_more": Bound("A", "B", 0, None)})
        with self.assertRaises(ValueError):
            capped.apply_bounds({"b0": Bound("A", "B", 0, 2),
                                 "extra": Bound("A", "B", 0, None)})
        with self.assertRaises(ValueError):
            capped.apply_bounds({"eq": Bound("A", "B", 1, 1)})
        # Withdrawing endpoints makes room again.
        capped.apply_bounds({"b0": Bound("A", "B", 0, None)})
        capped.apply_bounds({"one_more": Bound("A", "B", 0, None)})

    def test_max_envelope_functional(self):
        # 8 nodes, 40 observations, 10 finite endpoints, hand-known answer:
        # consistent unit observations on a path force x_i = i + const.
        nodes = [f"N{i}" for i in range(BOUNDED_MAX_NODES)]
        ledger, _, analyzer = system(nodes)
        batch = {f"o{i}:{k}": Observation(nodes[i], nodes[i + 1], 1, 1)
                 for i in range(7) for k in range(5)}
        batch.update({f"pad{k}": Observation("N0", "N1", 1, 1)
                      for k in range(BOUNDED_MAX_OBSERVATIONS - len(batch))})
        self.assertEqual(len(batch), BOUNDED_MAX_OBSERVATIONS)
        ledger.apply(batch)
        ledger.apply_bounds({f"b{i}": Bound(nodes[i], nodes[i + 4], 3, 5)
                             for i in range(4)} | {"b4": Bound("N0", "N7", 6, 8)})
        self.assertEqual(analyzer.range("N0", "N7"), (F(7), F(7)))
        self.assertEqual(analyzer.range("N7", "N0"), (F(-7), F(-7)))
        self.assertEqual(analyzer.range("N0", "N0"), (F(0), F(0)))
        # query keeps the unconstrained semantics: path variances add.
        self.assertEqual(analyzer.query("N0", "N7"), (F(7), F(13, 10)))

    def test_range_name_and_unknown_validation(self):
        ledger, _, analyzer = system(list("ABC"))
        ledger.apply_bounds({"ab": Bound("A", "B", 0, 1)})
        for a, b in (("Z", "Z"), ("A", "Z"), ("", "B"), (None, "A"), (1, "B")):
            with self.assertRaises(ValueError, msg=(a, b)):
                analyzer.range(a, b)
        self.assertEqual(analyzer.range("C", "C"), (F(0), F(0)))

    def test_query_and_current_ignore_bounds(self):
        # Bounds must not change the unconstrained query/current semantics.
        ledger, calibrator, analyzer = system(list("ABC"))
        ledger.apply({"ab": Observation("A", "B", 0, 1),
                      "ac": Observation("A", "C", 0, 1)})
        plain = calibrator.current()
        ledger.apply_bounds({"gap": Bound("C", "B", 2, None)})
        bounded = calibrator.current()
        self.assertEqual(bounded.revision, ledger.revision)
        self.assertEqual(plain.fits[0].values, bounded.fits[0].values)
        self.assertEqual(plain.fits[0].covariance, bounded.fits[0].covariance)
        self.assertEqual(analyzer.query("A", "B"), (F(0), F(1)))
        self.assertEqual(analyzer.query("B", "C"), (F(0), F(2)))
        with self.assertRaises(ValueError):
            analyzer.query("A", "Z")

    def test_snapshot_and_optimum_immutability(self):
        ledger, calibrator, analyzer = system(list("ABC"))
        ledger.apply({"ab": Observation("A", "B", 2, 1)})
        ledger.apply_bounds({"bc": Bound("B", "C", 0, 3)})
        snapshot = ledger.snapshot()
        optimum = calibrator.optimum()
        self.assertEqual(optimum.revision, ledger.revision)
        with self.assertRaises(TypeError):
            ledger.bounds["x"] = None
        with self.assertRaises(FrozenInstanceError):
            optimum.point = ()
        ledger.apply_bounds({"bc": Bound("B", "C", 0, 1)})
        ledger.apply({"ab": Observation("A", "B", 4, 1)})
        self.assertEqual(snapshot.rows, (("ab", Observation("A", "B", 2, 1)),))
        self.assertEqual(optimum.revision, 2)
        self.assertNotEqual(calibrator.optimum().revision, optimum.revision)
        self.assertEqual(analyzer.range("A", "C"), (F(4), F(5)))

    def test_revision_linkage_of_existing_analyzer(self):
        ledger, calibrator, analyzer = system(list("ABC"))
        ledger.apply({"ab": Observation("A", "B", 2, 1)})
        ledger.apply_bounds({"bc": Bound("B", "C", 0, 3)})
        self.assertEqual(analyzer.range("A", "C"), (F(2), F(5)))
        # Bound correction is visible without rebuilding the analyzer.
        ledger.apply_bounds({"bc": Bound("B", "C", 1, 2)})
        self.assertEqual(analyzer.range("A", "C"), (F(3), F(4)))
        # Observation correction is visible in the same revision semantics.
        ledger.apply({"ab": Observation("A", "B", 4, 1)})
        self.assertEqual(analyzer.range("A", "C"), (F(5), F(6)))
        # Withdrawal of every bound restores the unconstrained picture.
        ledger.apply_bounds({"bc": None})
        self.assertEqual(analyzer.range("A", "C"), (None, None))
        self.assertEqual(analyzer.range("A", "B"), (F(4), F(4)))
        self.assertEqual(calibrator.optimum().revision, ledger.revision)


# ---------------------------------------------------------------------------
# Randomized differential testing against the independent FM oracle.
# ---------------------------------------------------------------------------

class Differential(unittest.TestCase):
    def test_random_instances_against_oracle(self):
        rng = random.Random(20261116)
        for trial in range(40):
            n = rng.randint(1, 5)
            nodes = [f"N{i}" for i in range(n)]
            ledger, calibrator, analyzer = system(nodes)
            observations = {}
            if n >= 2:
                for k in range(rng.randint(0, 5)):
                    u, v = rng.sample(nodes, 2)
                    observations[f"o{k}"] = Observation(
                        u, v,
                        F(rng.randrange(-5, 6), rng.randrange(1, 4)),
                        F(rng.randrange(1, 4), rng.randrange(1, 3)))
            ledger.apply(observations)
            # Feasible-by-construction bounds around random potentials.
            potential = {node: F(rng.randrange(-4, 5)) for node in nodes}
            bounds = {}
            endpoints = 0
            if n >= 2:
                for k in range(rng.randint(0, 4)):
                    u, v = rng.sample(nodes, 2)
                    gap = potential[v] - potential[u]
                    lower = None if rng.randrange(3) == 0 else gap - rng.randrange(0, 4)
                    upper = None if rng.randrange(3) == 0 else gap + rng.randrange(0, 4)
                    if lower is None and upper is None:
                        upper = gap
                    added = (lower is not None) + (upper is not None)
                    if endpoints + added > BOUNDED_MAX_ENDPOINTS:
                        break
                    endpoints += added
                    bounds[f"b{k}"] = Bound(u, v, lower, upper)
            ledger.apply_bounds(bounds)
            rows = [obs for _, obs in ledger.snapshot().rows]
            current_bounds = [b for _, b in sorted(ledger.bounds.items())]
            optimum = calibrator.optimum()
            self.assertEqual(optimum.revision, ledger.revision)
            for a in nodes:
                for b in nodes:
                    expected = (F(0), F(0)) if a == b else oracle_range(
                        nodes, rows, current_bounds, a, b, optimum)
                    self.assertEqual(analyzer.range(a, b), expected,
                                     (trial, a, b))

    def test_random_corrections_against_oracle(self):
        rng = random.Random(424242)
        nodes = list("ABCD")
        ledger, calibrator, analyzer = system(nodes)
        ledger.apply({"o0": Observation("A", "B", 1, 1)})
        for step in range(12):
            if rng.randrange(2):
                ledger.apply({f"o{rng.randrange(4)}": Observation(
                    *rng.sample(nodes, 2),
                    F(rng.randrange(-4, 5)), F(rng.randrange(1, 4)))})
            else:
                # Feasible-by-construction bounds around random potentials;
                # a fixed set of bound IDs keeps the endpoint count bounded.
                potential = {node: F(rng.randrange(-3, 4)) for node in nodes}
                batch = {}
                for k in range(rng.randrange(1, 4)):
                    u, v = rng.sample(nodes, 2)
                    gap = potential[v] - potential[u]
                    batch[f"b{k}"] = Bound(u, v, gap - rng.randrange(0, 2),
                                           gap + rng.randrange(0, 2))
                ledger.apply_bounds(batch)
            rows = [obs for _, obs in ledger.snapshot().rows]
            current_bounds = [b for _, b in sorted(ledger.bounds.items())]
            optimum = calibrator.optimum()
            for a, b in (("A", "C"), ("C", "A"), ("B", "D"), ("A", "A")):
                expected = (F(0), F(0)) if a == b else oracle_range(
                    nodes, rows, current_bounds, a, b, optimum)
                self.assertEqual(analyzer.range(a, b), expected, (step, a, b))

    def test_feasibility_gate_matches_fm_oracle(self):
        rng = random.Random(7717)
        for trial in range(150):
            n = rng.randint(2, 5)
            nodes = [f"N{i}" for i in range(n)]
            ledger, _, _ = system(nodes)
            batch = {}
            for k in range(rng.randint(1, 4)):
                u, v = rng.sample(nodes, 2)
                lower = None if rng.randrange(3) == 0 else F(rng.randrange(-4, 3))
                upper = None if rng.randrange(3) == 0 else F(rng.randrange(-2, 5))
                if lower is None and upper is None:
                    lower = F(0)
                if lower is not None and upper is not None and lower > upper:
                    lower, upper = upper, lower
                batch[f"b{k}"] = Bound(u, v, lower, upper)
            candidate = list(batch.values())
            feasible = fm_feasible(n, difference_constraints(nodes, candidate))
            revision = ledger.revision
            if feasible:
                ledger.apply_bounds(batch)
                self.assertEqual(ledger.revision, revision + 1)
            else:
                with self.assertRaises(ValueError):
                    ledger.apply_bounds(batch)
                self.assertEqual(ledger.revision, revision)
                self.assertEqual(dict(ledger.bounds), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
