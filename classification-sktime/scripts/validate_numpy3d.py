#!/usr/bin/env python3
"""Validate basic sktime numpy3D classification arrays."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _load_labels(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    with path.open(newline="") as f:
        rows = [row for row in csv.reader(f) if row]
    if not rows:
        raise ValueError("label file is empty")
    if len(rows[0]) == 1:
        return np.asarray([row[0] for row in rows])
    if len(rows) == 1:
        return np.asarray(rows[0])
    raise ValueError("CSV labels must be one column or one row")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate sktime numpy3D X shape and y alignment."
    )
    parser.add_argument("x_npy", type=Path, help="Path to X .npy array")
    parser.add_argument("y", type=Path, help="Path to y .npy or simple CSV labels")
    args = parser.parse_args()

    X = np.load(args.x_npy, allow_pickle=False)
    y = _load_labels(args.y)

    errors: list[str] = []
    if X.ndim != 3:
        errors.append(f"X must be 3D (n_samples, n_channels, series_length), got {X.shape}")
    else:
        n_samples, n_channels, series_length = X.shape
        if n_samples <= 0:
            errors.append("X has zero samples")
        if n_channels <= 0:
            errors.append("X has zero channels")
        if series_length <= 0:
            errors.append("X has zero time points")
    if y.ndim != 1:
        errors.append(f"y must be 1D, got shape {y.shape}")
    if X.ndim >= 1 and y.ndim == 1 and len(y) != X.shape[0]:
        errors.append(f"len(y)={len(y)} must equal n_samples={X.shape[0]}")
    if np.issubdtype(X.dtype, np.number) and not np.isfinite(X).all():
        errors.append("X contains NaN or infinite numeric values")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    classes, counts = np.unique(y, return_counts=True)
    print("OK: valid basic sktime numpy3D classification input")
    print(f"X.shape={X.shape}")
    print(f"classes={dict(zip(classes.tolist(), counts.tolist()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
