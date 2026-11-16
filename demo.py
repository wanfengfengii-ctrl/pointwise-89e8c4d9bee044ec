"""Replay a finite public sample through the actual ledger and analyzer."""
import argparse
import json
from pathlib import Path
from ledger import Ledger, Observation
from calibration import Calibrator
from analysis import Analyzer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("sample")
    args = parser.parse_args()
    sample = json.loads(Path(args.sample).read_text(encoding="utf-8"))
    ledger = Ledger(sample["nodes"])
    analyzer = Analyzer(Calibrator(ledger))
    for stage in sample["stages"]:
        ledger.apply({key: None if value is None else Observation(*value)
                      for key, value in stage["batch"].items()})
        output = []
        for a, b in sample["queries"]:
            result = analyzer.query(a, b)
            output.append({"a": a, "b": b, "estimable": result is not None,
                           "estimate": None if result is None else str(result[0]),
                           "variance": None if result is None else str(result[1])})
        print(json.dumps({"stage": stage["name"], "revision": ledger.revision,
                          "comparisons": output}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
