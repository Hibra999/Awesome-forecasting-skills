#!/usr/bin/env python3
"""Validate THUML Time-Series-Library anomaly-detection inputs.

This script uses only the standard library. It checks official dataset file
presence, CSV numeric labels where applicable, .npy headers for dimensions,
and common run.py anomaly_detection arguments before using anomaly intervals
as changepoint proxies.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
from pathlib import Path


DATASET_FILES = {
    "PSM": ("train.csv", "test.csv", "test_label.csv"),
    "MSL": ("MSL_train.npy", "MSL_test.npy", "MSL_test_label.npy"),
    "SMAP": ("SMAP_train.npy", "SMAP_test.npy", "SMAP_test_label.npy"),
    "SMD": ("SMD_train.npy", "SMD_test.npy", "SMD_test_label.npy"),
    "SWAT": ("swat_train2.csv", "swat2.csv"),
}


def finite_number(raw: str) -> bool:
    try:
        return math.isfinite(float(raw))
    except (TypeError, ValueError):
        return False


def read_csv_shape(path: Path) -> tuple[int, int]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"{path}: missing header row")
        rows = sum(1 for _ in reader)
    return rows, len(header)


def validate_binary_csv_label(path: Path, skip_first_column: bool) -> int:
    positives = 0
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise SystemExit(f"{path}: missing header row")
        for row_number, row in enumerate(reader, start=2):
            values = row[1:] if skip_first_column else row[-1:]
            for raw in values:
                if not finite_number(raw):
                    raise SystemExit(f"{path}: row {row_number} label is not numeric")
                value = float(raw)
                if value not in {0.0, 1.0}:
                    raise SystemExit(f"{path}: row {row_number} label must be 0 or 1")
                positives += int(value == 1.0)
    return positives


def read_npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise SystemExit(f"{path}: not a .npy file")
        major = handle.read(1)[0]
        minor = handle.read(1)[0]
        if major == 1:
            header_len = struct.unpack("<H", handle.read(2))[0]
        elif major in {2, 3}:
            header_len = struct.unpack("<I", handle.read(4))[0]
        else:
            raise SystemExit(f"{path}: unsupported npy version {major}.{minor}")
        header = handle.read(header_len).decode("latin1")
    marker = "'shape':"
    start = header.find(marker)
    if start == -1:
        raise SystemExit(f"{path}: cannot find shape in npy header")
    start = header.find("(", start)
    end = header.find(")", start)
    if start == -1 or end == -1:
        raise SystemExit(f"{path}: malformed shape in npy header")
    pieces = [piece.strip() for piece in header[start + 1:end].split(",") if piece.strip()]
    return tuple(int(piece) for piece in pieces)


def validate_npy_dataset(root: Path, dataset: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    train = read_npy_shape(root / f"{dataset}_train.npy")
    test = read_npy_shape(root / f"{dataset}_test.npy")
    labels = read_npy_shape(root / f"{dataset}_test_label.npy")
    if len(train) != 2 or len(test) != 2:
        raise SystemExit(f"{dataset}: train/test arrays should be 2D [time, channels]")
    if train[1] != test[1]:
        raise SystemExit(f"{dataset}: train/test channel counts differ")
    if labels[0] != test[0]:
        raise SystemExit(f"{dataset}: test labels length must match test rows")
    return train, test, labels


def validate_args(args: argparse.Namespace) -> None:
    if args.dataset not in DATASET_FILES:
        raise SystemExit(f"--dataset must be one of: {', '.join(DATASET_FILES)}")
    if args.seq_len <= 1:
        raise SystemExit("--seq-len must be greater than 1")
    if args.anomaly_ratio <= 0 or args.anomaly_ratio >= 100:
        raise SystemExit("--anomaly-ratio must be in (0, 100)")
    if args.features != "M":
        raise SystemExit("official anomaly scripts use --features M")
    if args.pred_len != 0:
        raise SystemExit("official anomaly scripts use --pred-len 0")
    if args.enc_in <= 0 or args.c_out <= 0:
        raise SystemExit("--enc-in and --c-out must be positive")
    if args.enc_in != args.c_out:
        raise SystemExit("official anomaly reconstruction expects enc_in == c_out")


def validate_dataset(args: argparse.Namespace) -> int:
    validate_args(args)
    root = Path(args.root_path)
    missing = [name for name in DATASET_FILES[args.dataset] if not (root / name).exists()]
    if missing:
        raise SystemExit(
            f"{args.dataset}: missing files in {root}: {', '.join(missing)}"
        )

    if args.dataset == "PSM":
        train_rows, train_cols = read_csv_shape(root / "train.csv")
        test_rows, test_cols = read_csv_shape(root / "test.csv")
        label_rows, label_cols = read_csv_shape(root / "test_label.csv")
        if train_cols != test_cols:
            raise SystemExit("PSM: train.csv and test.csv column counts differ")
        if label_rows != test_rows:
            raise SystemExit("PSM: test_label.csv rows must match test.csv rows")
        channels = train_cols - 1
        positives = validate_binary_csv_label(root / "test_label.csv", True)
        train_shape = (train_rows, channels)
        test_shape = (test_rows, channels)
        label_shape = (label_rows, label_cols - 1)
    elif args.dataset == "SWAT":
        train_rows, train_cols = read_csv_shape(root / "swat_train2.csv")
        test_rows, test_cols = read_csv_shape(root / "swat2.csv")
        if train_cols != test_cols:
            raise SystemExit("SWAT: train/test column counts differ")
        channels = train_cols - 1
        positives = validate_binary_csv_label(root / "swat2.csv", False)
        train_shape = (train_rows, channels)
        test_shape = (test_rows, channels)
        label_shape = (test_rows, 1)
    else:
        train_shape, test_shape, label_shape = validate_npy_dataset(root, args.dataset)
        channels = train_shape[1]
        positives = -1

    if args.seq_len >= train_shape[0] or args.seq_len >= test_shape[0]:
        raise SystemExit("--seq-len must be smaller than train and test row counts")
    if args.enc_in != channels or args.c_out != channels:
        raise SystemExit(
            f"--enc-in/--c-out must match channel count {channels} for {args.dataset}"
        )

    msg = (
        "OK tslib anomaly input: "
        f"dataset={args.dataset} train={train_shape} test={test_shape} "
        f"labels={label_shape} seq_len={args.seq_len} channels={channels}"
    )
    if positives >= 0:
        msg += f" positive_labels={positives}"
    print(msg)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate THUML Time-Series-Library anomaly-detection files."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--root-path", required=True)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--enc-in", type=int, required=True)
    parser.add_argument("--c-out", type=int, required=True)
    parser.add_argument("--features", default="M")
    parser.add_argument("--pred-len", type=int, default=0)
    parser.add_argument("--anomaly-ratio", type=float, default=1.0)
    return parser


def main() -> int:
    return validate_dataset(build_parser().parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
