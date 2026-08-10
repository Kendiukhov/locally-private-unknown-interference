#!/usr/bin/env python3
"""Finalize metadata/checksums after running experiments through separate entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lpui.reproducibility import write_reproducibility_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/experiments.json"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    with args.config.open(encoding="utf-8") as stream:
        configuration = json.load(stream)
    project_root = Path(__file__).resolve().parents[1]
    metadata_path, checksum_path = write_reproducibility_files(
        output=args.output,
        project_root=project_root,
        config_path=args.config,
        configuration=configuration,
        quick=False,
        skip_doi=False,
    )
    print(f"wrote {metadata_path} and {checksum_path}")


if __name__ == "__main__":
    main()
