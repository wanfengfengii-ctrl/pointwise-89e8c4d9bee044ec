"""Single entry running the original regression plus the bounded verification."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

STEPS = (
    ("original regression", [sys.executable, "-B", "checks.py"]),
    ("public bound examples", [sys.executable, "-B", "bounds_examples.py", "--verify"]),
    ("independent bound verification", [sys.executable, "-B", "bounds_checks.py"]),
)


def main():
    for label, argv in STEPS:
        print(f"+ {label}: {' '.join(argv)}", flush=True)
        result = subprocess.run(argv, cwd=HERE)
        if result.returncode != 0:
            print(f"FAILED: {label}", flush=True)
            return result.returncode
    print("all verification steps passed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
