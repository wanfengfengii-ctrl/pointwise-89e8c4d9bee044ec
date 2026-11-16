"""Public hand-calculated examples for the bounded-calibration extension."""
import argparse
import json
from fractions import Fraction as F


def exercise():
    from ledger import Ledger, Observation as O, Bound
    from calibration import Calibrator
    from analysis import Analyzer

    events = []

    def record(label, ledger, analyzer, pair, expected):
        result = analyzer.range(*pair)
        assert type(result) is tuple and len(result) == 2
        assert all(v is None or type(v) is F for v in result)
        assert result == expected, (label, result, expected)
        events.append({
            "event": label, "revision": ledger.revision,
            "pair": list(pair),
            "range": [None if v is None else str(v) for v in result],
        })

    def reject(action):
        try:
            action()
        except ValueError:
            return
        raise AssertionError("expected ValueError")

    ledger = Ledger(list("ABC"))
    calibrator = Calibrator(ledger)
    analyzer = Analyzer(calibrator)
    ledger.apply({"ab": O("A", "B", 0, 1), "ac": O("A", "C", 0, 1)})
    old_snapshot = ledger.snapshot()
    old_state = calibrator.current()
    revision = ledger.revision
    assert ledger.apply_bounds({"separation": Bound("C", "B", 2, None)}) == revision + 1
    record("joint_B", ledger, analyzer, ("A", "B"), (F(1), F(1)))
    record("joint_C", ledger, analyzer, ("A", "C"), (F(-1), F(-1)))
    assert analyzer.query("A", "B") == (F(0), F(1))
    assert old_snapshot.revision == revision and old_state.revision == revision
    assert old_state.fits[0].values == (F(0), F(0), F(0))
    assert ledger.snapshot().rows == old_snapshot.rows

    ledger = Ledger(list("ABC"))
    analyzer = Analyzer(Calibrator(ledger))
    ledger.apply({"ab": O("A", "B", 2, 1)})
    # Observation and bound identifiers are deliberately identical.
    ledger.apply_bounds({"ab": Bound("B", "C", 0, 3)})
    record("finite", ledger, analyzer, ("A", "C"), (F(2), F(5)))
    record("reverse", ledger, analyzer, ("C", "A"), (F(-5), F(-2)))
    assert analyzer.query("A", "C") is None
    revision, snapshot = ledger.revision, ledger.snapshot()
    reject(lambda: ledger.apply_bounds({
        "ab": Bound("B", "C", 0, 3),
        "impossible": Bound("B", "C", None, -1),
    }))
    assert ledger.revision == revision and ledger.snapshot() == snapshot
    record("rejected_batch_unchanged", ledger, analyzer, ("A", "C"), (F(2), F(5)))
    ledger.apply({"ab": O("A", "B", 4, 1)})
    record("observation_corrected", ledger, analyzer, ("A", "C"), (F(4), F(7)))
    ledger.apply_bounds({"ab": Bound("B", "C", 0, None)})
    record("one_sided", ledger, analyzer, ("A", "C"), (F(4), None))
    ledger.apply_bounds({"ab": None})
    record("withdrawn", ledger, analyzer, ("A", "C"), (None, None))
    record("self", ledger, analyzer, ("C", "C"), (F(0), F(0)))
    reject(lambda: analyzer.range("unknown", "unknown"))
    revision = ledger.revision
    ledger.apply_bounds({})
    assert ledger.revision == revision + 1
    reject(lambda: ledger.apply_bounds({"unknown": None}))

    ledger = Ledger(list("ABC"))
    analyzer = Analyzer(Calibrator(ledger))
    ledger.apply_bounds({
        "ab": Bound("A", "B", 0, 2),
        "bc": Bound("B", "C", 1, 3),
    })
    record("no_observations", ledger, analyzer, ("A", "C"), (F(1), F(5)))
    before = ledger.snapshot()
    reject(lambda: ledger.apply_bounds({"ac": Bound("A", "C", None, 0)}))
    assert ledger.snapshot() == before
    record("negative_cycle_rejected", ledger, analyzer, ("A", "C"), (F(1), F(5)))

    ledger = Ledger(list("ABC"))
    analyzer = Analyzer(Calibrator(ledger))
    ledger.apply({"ab": O("A", "B", 0, 1), "ac": O("A", "C", 0, 1)})
    ledger.apply_bounds({
        "equal": Bound("C", "B", "2", F(2)),
        "redundant": Bound("C", "B", 2, None),
    })
    record("equality_and_redundancy", ledger, analyzer, ("A", "B"), (F(1), F(1)))
    return events


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--verify", action="store_true")
    modes.add_argument("--demo", action="store_true")
    args = parser.parse_args()
    events = exercise()
    if args.demo:
        for event in events:
            print(json.dumps(event, ensure_ascii=False, sort_keys=True))
    else:
        print(json.dumps({"verified_examples": len(events),
                          "last_result": events[-1]}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
