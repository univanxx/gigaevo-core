#!/usr/bin/env python3
"""
Run two Heilbron programs and compare their outputs (fitness, is_valid, etc.).

Usage (from repo root):
  python -m tools.compare_programs \\
    --program-a problems/heilbron/initial_programs/random_arr.py \\
    --program-b path/to/best_evolved.py

  python -m tools.compare_programs \\
    --program-a problems/heilbron/initial_programs/random_arr.py \\
    --program-b path/to/best_evolved.py \\
    --runs 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root = parent of tools/
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HEILBRON_DIR = REPO_ROOT / "problems" / "heilbron"


def _run_program(program_path: Path, heilbron_dir: Path, *, runs: int = 1) -> list[dict]:
    """Load program, call entrypoint(), validate(); return list of result dicts."""
    program_path = program_path.resolve()
    if not program_path.exists():
        raise FileNotFoundError(program_path)

    results = []
    import importlib.util

    for run_idx in range(runs):
        # Fresh module each run so top-level seed() is applied again
        spec = importlib.util.spec_from_file_location(
            f"user_prog_{run_idx}",
            program_path,
        )
        mod = importlib.util.module_from_spec(spec)
        sys.path.insert(0, str(heilbron_dir))
        try:
            spec.loader.exec_module(mod)
        finally:
            sys.path.pop(0)

        if not hasattr(mod, "entrypoint"):
            results.append({"error": "No entrypoint() in module"})
            continue

        try:
            points = mod.entrypoint()
        except Exception as e:
            results.append({"error": f"entrypoint() failed: {e}"})
            continue

        # Validate using problem's validate()
        sys.path.insert(0, str(heilbron_dir))
        try:
            from validate import validate
            out = validate(points)
            results.append(out)
        except Exception as e:
            results.append({"error": f"validate() failed: {e}"})
        finally:
            sys.path.pop(0)

    return results


def main():
    parser = argparse.ArgumentParser(description="Run and compare two Heilbron programs")
    parser.add_argument("--program-a", type=Path, required=True, help="First program .py path")
    parser.add_argument("--program-b", type=Path, required=True, help="Second program .py path")
    parser.add_argument("--runs", type=int, default=1, help="Run each program N times (for stochastic code)")
    parser.add_argument(
        "--problem-dir",
        type=Path,
        default=DEFAULT_HEILBRON_DIR,
        help="Problem dir (default: problems/heilbron)",
    )
    args = parser.parse_args()

    heilbron_dir = args.problem_dir.resolve()

    a_results = _run_program(args.program_a.resolve(), heilbron_dir, runs=args.runs)
    b_results = _run_program(args.program_b.resolve(), heilbron_dir, runs=args.runs)

    def has_errors(res_list):
        return any("error" in r for r in res_list)

    def fitness_list(res_list):
        return [r["fitness"] for r in res_list if "error" not in r]

    print("=" * 60)
    print("Program A:", args.program_a)
    print("Program B:", args.program_b)
    print("Runs:", args.runs)
    print("=" * 60)

    if has_errors(a_results):
        print("\nProgram A errors:")
        for i, r in enumerate(a_results):
            if "error" in r:
                print(f"  run {i+1}: {r['error']}")
    else:
        fa = fitness_list(a_results)
        print(f"\nProgram A — fitness: min={min(fa):.6f}, max={max(fa):.6f}", end="")
        if args.runs > 1:
            import statistics
            print(f", mean={statistics.mean(fa):.6f}, stdev={statistics.stdev(fa):.6f}")
        else:
            print(f"  (is_valid={a_results[0].get('is_valid', 'N/A')})")

    if has_errors(b_results):
        print("\nProgram B errors:")
        for i, r in enumerate(b_results):
            if "error" in r:
                print(f"  run {i+1}: {r['error']}")
    else:
        fb = fitness_list(b_results)
        print(f"\nProgram B — fitness: min={min(fb):.6f}, max={max(fb):.6f}", end="")
        if args.runs > 1:
            import statistics
            print(f", mean={statistics.mean(fb):.6f}, stdev={statistics.stdev(fb):.6f}")
        else:
            print(f"  (is_valid={b_results[0].get('is_valid', 'N/A')})")

    if not has_errors(a_results) and not has_errors(b_results):
        ma = sum(fitness_list(a_results)) / len(fitness_list(a_results))
        mb = sum(fitness_list(b_results)) / len(fitness_list(b_results))
        print(f"\nComparison: B vs A fitness ratio = {mb/ma:.4f} (B better if > 1)")
    print()


if __name__ == "__main__":
    main()
