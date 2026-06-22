#!/usr/bin/env python3
"""Basic validation for Time-Series-Library UEA TRAIN/TEST .ts files."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def _read_ts(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace").splitlines()
    metadata: list[str] = []
    data_lines: list[str] = []
    in_data = False
    for raw in text:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("@data"):
            in_data = True
            continue
        if not in_data and line.startswith("@"):
            metadata.append(line)
            continue
        if in_data:
            data_lines.append(line)

    class_meta = [line for line in metadata if line.lower().startswith("@classlabel")]
    labels = []
    for line in data_lines:
        parts = line.rsplit(":", 1)
        if len(parts) == 2 and parts[-1].strip():
            labels.append(parts[-1].strip())

    return {
        "metadata": metadata,
        "has_data_tag": in_data,
        "n_rows": len(data_lines),
        "class_meta": class_meta,
        "labels": labels,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate required UEA .ts split files for THUML Time-Series-Library classification."
    )
    parser.add_argument("root_path", type=Path, help="Dataset directory, e.g. ./dataset/Heartbeat")
    parser.add_argument("dataset_name", help="Dataset/model_id, e.g. Heartbeat")
    args = parser.parse_args()

    errors: list[str] = []
    summaries = {}
    for split in ("TRAIN", "TEST"):
        path = args.root_path / f"{args.dataset_name}_{split}.ts"
        try:
            summary = _read_ts(path)
        except Exception as exc:
            errors.append(f"{split}: {exc}")
            continue
        summaries[split] = summary
        if not summary["has_data_tag"]:
            errors.append(f"{split}: missing @data section")
        if summary["n_rows"] == 0:
            errors.append(f"{split}: no data rows")
        if not summary["class_meta"]:
            errors.append(f"{split}: missing @classLabel metadata")
        if len(summary["labels"]) != summary["n_rows"]:
            errors.append(f"{split}: could not parse one class label per data row")

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    for split, summary in summaries.items():
        counts = Counter(summary["labels"])
        print(f"{split}: rows={summary['n_rows']} classes={dict(sorted(counts.items()))}")
    print("OK: required UEA TRAIN/TEST .ts files are present and structurally plausible")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
