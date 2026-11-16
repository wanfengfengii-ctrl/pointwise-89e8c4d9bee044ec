"""Deterministic public regressions and an independent singular-system oracle."""
import random
import unittest
from dataclasses import FrozenInstanceError
from fractions import Fraction as F
from ledger import Ledger, Observation as O
from calibration import Calibrator
from analysis import Analyzer
from reference import Reference
from workloads import workload


def solve_singular(matrix, rhs):
    """RREF with free coordinates zero; inconsistent equations return None."""
    n = len(rhs)
    rows = [list(row) + [value] for row, value in zip(matrix, rhs)]
    pivots = []
    lead = 0
    for col in range(n):
        pivot = next((r for r in range(lead, n) if rows[r][col]), None)
        if pivot is None:
            continue
        rows[lead], rows[pivot] = rows[pivot], rows[lead]
        p = rows[lead][col]
        rows[lead] = [v / p for v in rows[lead]]
        for r in range(n):
            if r != lead:
                scale = rows[r][col]
                rows[r] = [x - scale * y for x, y in zip(rows[r], rows[lead])]
        pivots.append(col)
        lead += 1
    if any(not any(row[:n]) and row[n] for row in rows):
        return None
    solution = [F(0)] * n
    for r, col in enumerate(pivots):
        solution[col] = rows[r][n]
    return solution


def oracle(nodes, observations, a, b):
    # Build from dense incidence vectors; neither components nor grounded inverse.
    n = len(nodes)
    matrix = [[F(0) for _ in nodes] for _ in nodes]
    rhs = [F(0)] * n
    for obs in observations.values():
        row = [F(int(node == obs.v) - int(node == obs.u)) for node in nodes]
        for i in range(n):
            rhs[i] += obs.w * row[i] * obs.d
            for j in range(n):
                matrix[i][j] += obs.w * row[i] * row[j]
    contrast = [F(int(node == b) - int(node == a)) for node in nodes]
    variance_solution = solve_singular(matrix, contrast)
    if variance_solution is None:
        return None
    fitted = solve_singular(matrix, rhs)
    return (sum((x * y for x, y in zip(contrast, fitted)), F(0)),
            sum((x * y for x, y in zip(contrast, variance_solution)), F(0)))


def system(nodes):
    ledger = Ledger(nodes)
    calibrator = Calibrator(ledger)
    return ledger, calibrator, Analyzer(calibrator)


class Regression(unittest.TestCase):
    def test_bridge_retract_reconnect_and_weight(self):
        ledger, calibrator, analyzer = system(list("ABCD"))
        ledger.apply({"ab": O("A", "B", 2, 1), "bc": O("B", "C", 3, 1)})
        self.assertEqual(analyzer.query("A", "C"), (F(5), F(2)))
        before = calibrator.current()
        ledger.apply({"bc": None})
        self.assertIsNone(analyzer.query("A", "C"))
        self.assertEqual(analyzer.query("A", "B"), (F(2), F(1)))
        self.assertEqual(analyzer.query("C", "C"), (F(0), F(0)))
        self.assertIsNone(analyzer.query("A", "D"))
        self.assertEqual(before.revision, 1)
        self.assertEqual(before.membership["A"], before.membership["C"])
        ledger.apply({"bc": O("B", "C", 4, 2)})
        self.assertEqual(analyzer.query("A", "C"), (F(6), F(3, 2)))
        ledger.apply({"ab": O("A", "B", 2, 4)})
        self.assertEqual(analyzer.query("A", "B"), (F(2), F(1, 4)))
        self.assertEqual(analyzer.query("A", "C"), (F(6), F(3, 4)))
        self.assertEqual(calibrator.current().revision, ledger.revision)

    def test_parallel_inconsistent_cycle_and_endpoint_replacement(self):
        ledger, _, analyzer = system(["A", "B", "C", "D"])
        ledger.apply({"p": O("A", "B", 2, 1), "q": O("A", "B", 4, 3)})
        self.assertEqual(analyzer.query("A", "B"), (F(7, 2), F(1, 4)))
        ledger.apply({"q": None})
        self.assertEqual(analyzer.query("A", "B"), (F(2), F(1)))
        ledger.apply({"bc": O("B", "C", 3, 1), "ac": O("A", "C", 8, 1)})
        self.assertEqual(analyzer.query("A", "C"), (F(7), F(2, 3)))
        self.assertEqual(analyzer.query("A", "B"), (F(3), F(2, 3)))
        ledger.apply({"p": O("C", "D", -2, 2), "bc": None,
                      "ac": O("D", "A", -6, 4)})
        self.assertEqual(analyzer.query("C", "A"), (F(-8), F(3, 4)))
        self.assertIsNone(analyzer.query("A", "B"))

    def test_validation_and_atomicity(self):
        ledger, _, analyzer = system(["A", "B", "C"])
        ledger.apply({"ab": O("A", "B", 2, 1)})
        for batch in (
            {"ab": O("A", "B", 99, 3), "missing": None},
            {"ab": None, "bad": O("A", "Z", 1, 1)},
            {"ab": O("A", "B", 99, 3), "bad": ("A", "B", 1, 1)},
            {"ab": None, "": O("A", "B", 1, 1)},
            [("ab", None)],
        ):
            before = ledger.snapshot()
            with self.assertRaises(ValueError):
                ledger.apply(batch)
            self.assertEqual(ledger.snapshot(), before)
            self.assertEqual(analyzer.query("A", "B"), (F(2), F(1)))
        for args in (
            ("A", "A", 1, 1), ("", "B", 1, 1), ("A", "B", True, 1),
            ("A", "B", 1.2, 1), ("A", "B", "1/0", 1),
            ("A", "B", 1, 0), ("A", "B", 1, -1), ("A", "B", 1, True),
        ):
            with self.assertRaises(ValueError):
                O(*args)
        for nodes in (["A", "A"], ["A", ""], "AB", None):
            with self.assertRaises(ValueError):
                Ledger(nodes)
        for a, b in (("Z", "Z"), ("A", "Z"), ("", "B"), (None, "A")):
            with self.assertRaises(ValueError):
                analyzer.query(a, b)
        revision = ledger.revision
        ledger.apply({})
        self.assertEqual(ledger.revision, revision + 1)
        self.assertEqual(analyzer.query("B", "A"), (F(-2), F(1)))

    def test_snapshot_and_state_are_immutable(self):
        ledger, calibrator, analyzer = system(["A", "B"])
        obs = O("A", "B", "2/3", "3/5")
        ledger.apply({"x": obs})
        snapshot = ledger.snapshot()
        state = calibrator.current()
        with self.assertRaises(FrozenInstanceError):
            obs.w = F(9)
        with self.assertRaises(TypeError):
            state.membership["A"] = 99
        with self.assertRaises(TypeError):
            state.fits[0].positions["A"] = 99
        ledger.apply({"x": None})
        self.assertEqual(snapshot.rows, (("x", obs),))
        self.assertIsNone(analyzer.query("A", "B"))
        self.assertEqual(state.fits[0].values[1], F(2, 3))
        empty, _, empty_analyzer = system([])
        self.assertEqual(empty.snapshot().rows, ())
        with self.assertRaises(ValueError):
            empty_analyzer.query("A", "A")

    def test_orientation_and_relabeling(self):
        original = {"x": O("A", "B", "1/3", 2), "y": O("B", "C", 4, 3),
                    "z": O("A", "C", 6, "2/5")}
        rename = {"A": "z", "B": "m", "C": "a"}
        first, _, analysis = system(["A", "B", "C"])
        second, _, renamed = system(rename.values())
        first.apply(original)
        second.apply({key: O(rename[o.v], rename[o.u], -o.d, o.w)
                      for key, o in original.items()})
        for a in rename:
            for b in rename:
                self.assertEqual(analysis.query(a, b), renamed.query(rename[a], rename[b]))

    def test_fixed_seed_continuous_differential(self):
        nodes = tuple("ABCDEF")
        ledger, _, analyzer = system(nodes)
        reference = Reference(nodes)
        rows = {}
        rng = random.Random(71023)
        for step in range(50):
            batch = {}
            for key in rng.sample([f"id{i}" for i in range(12)], rng.randrange(1, 5)):
                if key in rows and rng.randrange(3) == 0:
                    batch[key] = None
                else:
                    u, v = rng.sample(nodes, 2)
                    batch[key] = O(u, v, F(rng.randrange(-7, 8), rng.randrange(1, 5)),
                                   F(rng.randrange(1, 5), rng.randrange(1, 4)))
            ledger.apply(batch)
            reference.ledger.apply(batch)
            for key, value in batch.items():
                if value is None:
                    del rows[key]
                else:
                    rows[key] = value
            for a in nodes:
                for b in nodes:
                    expected = oracle(nodes, rows, a, b)
                    self.assertEqual(analyzer.query(a, b), expected, (step, a, b))
                    self.assertEqual(reference.query(a, b), expected, (step, a, b))

    def test_frozen_workload_semantics(self):
        for kind in ("stable", "split"):
            data = workload(8, kind, 6)
            self.assertEqual(data, workload(8, kind, 6))
            nodes, initial, batches, pairs = data
            ledger, _, analyzer = system(nodes)
            reference = Reference(nodes)
            ledger.apply(initial)
            reference.ledger.apply(initial)
            for step, batch in enumerate(batches):
                ledger.apply(batch)
                reference.ledger.apply(batch)
                self.assertEqual([analyzer.query(*p) for p in pairs],
                                 [reference.query(*p) for p in pairs])
                cross = analyzer.query(nodes[0], nodes[-1])
                self.assertEqual(cross is None, kind == "split" and step % 2 == 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
