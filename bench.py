"""Measure whole correction/query cycles; no stored answer cache."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time
import tracemalloc
from ledger import Ledger
from calibration import Calibrator
from analysis import Analyzer
from reference import Reference
from workloads import workload


def target(nodes):
    ledger = Ledger(nodes)
    return ledger, Analyzer(Calibrator(ledger))


def reference(nodes):
    analyzer = Reference(nodes)
    return analyzer.ledger, analyzer


def initialize(factory, data):
    nodes, initial, _, pairs = data
    ledger, analyzer = factory(nodes)
    ledger.apply(initial)
    for pair in pairs:
        analyzer.query(*pair)
    return ledger, analyzer


def cycle(ledger, analyzer, data):
    results = []
    for batch in data[2]:
        ledger.apply(batch)
        results.append(tuple(analyzer.query(*pair) for pair in data[3]))
    return tuple(results)


def timed(factory, data):
    gc.collect()
    start = time.perf_counter()
    ledger, analyzer = initialize(factory, data)
    initialized = time.perf_counter() - start
    start = time.perf_counter()
    results = cycle(ledger, analyzer, data)
    elapsed = time.perf_counter() - start
    return initialized, elapsed, results


def peak(factory, data):
    gc.collect()
    tracemalloc.start()
    try:
        ledger, analyzer = initialize(factory, data)
        results = cycle(ledger, analyzer, data)
        return tracemalloc.get_traced_memory()[1], results
    finally:
        tracemalloc.stop()


def digest(results):
    serial = [[[None if result is None else [str(x) for x in result]
                for result in batch] for batch in results]]
    return hashlib.sha256(json.dumps(serial, separators=(",", ":")).encode()).hexdigest()


def measure(size, kind, rounds=3, batches=20):
    data = workload(size, kind, batches)
    times = {"target": [], "reference": []}
    init = {"target": [], "reference": []}
    expected = None
    for repeat in range(rounds):
        order = (("target", target), ("reference", reference))
        if repeat % 2:
            order = order[::-1]
        for label, factory in order:
            construction, elapsed, results = timed(factory, data)
            if expected is None:
                expected = results
            if results != expected:
                raise AssertionError(f"business result mismatch: {size}/{kind}/{label}")
            init[label].append(construction)
            times[label].append(elapsed)
    memory, results = peak(target, data)
    if results != expected:
        raise AssertionError("memory pass result mismatch")
    medians = {key: statistics.median(value) for key, value in times.items()}
    return {
        "size": size, "kind": kind, "batches": batches, "rounds": rounds,
        "seconds": times, "median_seconds": medians,
        "initialization_seconds": init,
        "target_to_reference": medians["target"] / medians["reference"],
        "target_peak_bytes_including_initialization": memory,
        "result_sha256": digest(expected),
        "within_goal": medians["target"] <= medians["reference"] / 4
                       and memory <= 96 * 1024 * 1024,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()
    sizes, rounds, batches = ((8,), 1, 4) if args.smoke else ((80, 160), 3, 20)
    sources = {path: hashlib.sha256(Path(__file__).with_name(path).read_bytes()).hexdigest()
               for path in ("reference.py", "workloads.py", "bench.py")}
    print(json.dumps({"sources_sha256": sources}, sort_keys=True), flush=True)
    passed = True
    for size in sizes:
        for kind in ("stable", "split"):
            result = measure(size, kind, rounds, batches)
            print(json.dumps(result, sort_keys=True), flush=True)
            passed = passed and result["within_goal"]
    if args.gate and not args.smoke and not passed:
        raise SystemExit(1)
    if args.gate and args.smoke:
        raise SystemExit("smoke is not a performance admission run")


if __name__ == "__main__":
    main()
