"""
run_all_tests.py - runs every test script of the project and prints one summary.

Run from the project folder:

    python tests/run_all_tests.py             # about 2 minutes
    python tests/run_all_tests.py --retrain   # also retrains the model from scratch and compares (about 1 minute more)
"""

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
retrain = ["--retrain"] if "--retrain" in sys.argv else []

SUITES = [
    ("Model artifacts and result files", "verify_models.py", retrain),
    ("Authentication and page protection", "test_auth.py", []),
    ("Batch prediction", "test_batch.py", []),
    ("Application: prediction, validation, pages", "test_app.py", []),
]

summary = []
for title, script, extra in SUITES:
    print(f"\n{'=' * 78}\n{title}   ({script})\n{'=' * 78}", flush=True)
    start = time.time()
    proc = subprocess.run([sys.executable, str(HERE / script)] + extra, cwd=PROJECT, capture_output=True, text=True, encoding="utf-8")
    lines = [l for l in proc.stdout.splitlines() if l.startswith("[") or l.startswith("SUMMARY") or l.startswith("===")]
    print("\n".join(lines))
    if proc.returncode != 0 and not any(l.startswith("SUMMARY") for l in lines):
        print(proc.stderr[-1500:])
    summary.append((title, proc.returncode == 0, time.time() - start))

print(f"\n{'=' * 78}\nOVERALL RESULT\n{'=' * 78}")
for title, ok, seconds in summary:
    print(f"  {'PASS' if ok else 'FAIL'}  {title}  ({seconds:.0f} s)")
all_ok = all(ok for _, ok, _ in summary)
print("\nALL TEST SUITES PASSED" if all_ok else "\nSOME TESTS FAILED - read the [FAIL] lines above")
sys.exit(0 if all_ok else 1)
