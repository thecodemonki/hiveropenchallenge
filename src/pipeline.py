"""End-to-end orchestration for the suggested-reply pipeline.

Runs dataset generation, reply generation, and evaluation in sequence.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent


def run_step(script: str, args: list[str]) -> None:
    cmd = [sys.executable, str(SRC / script), *args]
    print(f"\n=== {' '.join(cmd)} ===\n", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def run_pipeline(n: int = 40, dry_run: bool = False) -> None:
    dry = ["--dry-run"] if dry_run else []
    run_step("generate_dataset.py", ["--n", str(n), *dry])
    run_step("generate_reply.py", list(dry))
    run_step("evaluate.py", list(dry))
    print("\n=== pipeline complete ===\n", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=40, help="dataset size for generate_dataset")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="pass --dry-run through to every stage",
    )
    args = parser.parse_args()
    run_pipeline(n=args.n, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
