"""Single demo entry: original sample replay, then bounded-correction ranges."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = (
    ("original sample replay", [sys.executable, "-B", "demo.py", "samples/network.json"]),
    ("ranges across bound corrections", [sys.executable, "-B", "bounds_examples.py", "--demo"]),
)


def main():
    for label, argv in STEPS:
        print(f"+ {label}: {' '.join(argv)}", flush=True)
        result = subprocess.run(argv, cwd=HERE)
        if result.returncode != 0:
            print(f"FAILED: {label}", flush=True)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
