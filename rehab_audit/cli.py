from __future__ import annotations

import argparse
from pathlib import Path

from .acquire import fetch_files
from .pipeline import check, run_study, verify


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Independent REHAB record-aware signal audit")
    parser.add_argument("command", choices=("fetch", "run", "verify", "check"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    cache = args.cache or root / "data/cache"
    output = args.output or root / "results"
    try:
        if args.command == "fetch":
            fetch_files(root / "config/source_files.json", cache)
        elif args.command == "run":
            result = run_study(root, cache, output)
            print(f"Saved {len(result['runs'])} comparisons to {output}")
        elif args.command == "verify":
            verify(root, output)
            print("Source and result hashes verified")
        else:
            check(root, cache, output)
            print("Fresh replay revalidated inputs and matched all result files")
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"Audit stopped: {exc}\n")
    return 0
