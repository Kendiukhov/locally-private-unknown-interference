"""Provenance manifests for generated Monte Carlo artifacts."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import scipy


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_manifest(project_root: Path, config_path: Path) -> dict[str, str]:
    """Hash executable source and configuration without embedding local paths."""
    candidates = [
        project_root / "pyproject.toml",
        project_root / "requirements-lock.txt",
        project_root / "Makefile",
        config_path.resolve(),
    ]
    candidates.extend(sorted((project_root / "src").rglob("*.py")))
    candidates.extend(sorted((project_root / "scripts").rglob("*.py")))
    candidates.extend(sorted((project_root / "tests").rglob("*.py")))
    manifest: dict[str, str] = {}
    for path in candidates:
        if path.is_file() and not path.name.startswith("._"):
            try:
                label = str(path.relative_to(project_root))
            except ValueError:
                label = path.name
            manifest[label] = file_sha256(path)
    return dict(sorted(manifest.items()))


def write_reproducibility_files(
    *,
    output: Path,
    project_root: Path,
    config_path: Path,
    configuration: dict[str, Any],
    quick: bool,
    skip_doi: bool,
) -> tuple[Path, Path]:
    """Write runtime metadata and checksums for exactly the produced result grid."""
    metadata_path = output / "run_metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "configuration": configuration,
        "quick": quick,
        "skip_doi": skip_doi,
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "source_manifest": source_manifest(project_root, config_path),
    }
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    result_names = [
        "benchmark",
        "sample_size_scaling",
        "interference_scaling",
        "exposure_overlap",
    ]
    if not skip_doi:
        result_names.append("doi_sieve")
    result_paths = [
        output / directory / f"{name}.csv"
        for name in result_names
        for directory in ("raw", "summary")
    ]
    missing = [str(path) for path in result_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"cannot finalize missing result files: {missing}")
    checksums = {
        str(path.relative_to(output)): file_sha256(path)
        for path in sorted(result_paths)
        if not path.name.startswith("._")
    }
    checksum_path = output / "checksums.json"
    with checksum_path.open("w", encoding="utf-8") as stream:
        json.dump(checksums, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return metadata_path, checksum_path


def verify_result_checksums(output: Path) -> list[str]:
    """Return human-readable checksum failures; an empty list means success."""
    checksum_path = output / "checksums.json"
    with checksum_path.open(encoding="utf-8") as stream:
        expected = json.load(stream)
    failures: list[str] = []
    for relative_path, expected_digest in expected.items():
        path = output / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
        elif file_sha256(path) != expected_digest:
            failures.append(f"digest mismatch: {relative_path}")
    return failures
