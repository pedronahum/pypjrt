#!/usr/bin/env python3
"""Run every example and report. Used by CI so the examples cannot rot.

    python examples/run_all.py [plugin.so]
"""
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent


def main(argv: list[str]) -> int:
    scripts = sorted(p for p in HERE.glob("[0-9][0-9]_*.py"))
    if not scripts:
        print("no examples found", file=sys.stderr)
        return 1

    failures = []
    for script in scripts:
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, str(script), *argv],
                              capture_output=True, text=True)
        elapsed = time.perf_counter() - t0
        ok = proc.returncode == 0
        print(f"{'ok  ' if ok else 'FAIL'} {script.name:<34} {elapsed:6.2f}s")
        if not ok:
            failures.append((script.name, proc.stdout[-1500:], proc.stderr[-1500:]))

    print(f"\n{len(scripts) - len(failures)}/{len(scripts)} examples ran clean")
    for name, out, err in failures:
        print(f"\n--- {name} ---\n{out}\n{err}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
