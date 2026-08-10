#!/usr/bin/env python3
"""Verify stored result digests and the minimum manuscript reference count."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from lpui.reproducibility import verify_result_checksums


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--aux", type=Path, default=Path("paper/main.aux"))
    args = parser.parse_args()

    failures = verify_result_checksums(args.results)
    if failures:
        raise SystemExit("\n".join(failures))
    cited_keys: set[str] = set()
    if args.aux.exists():
        for group in re.findall(r"\\citation\{([^}]*)\}", args.aux.read_text()):
            cited_keys.update(key for key in group.split(",") if key)
        if len(cited_keys) < 30:
            raise SystemExit(f"manuscript cites only {len(cited_keys)} unique sources")
    print(f"verified result checksums and {len(cited_keys)} cited sources")


if __name__ == "__main__":
    main()
