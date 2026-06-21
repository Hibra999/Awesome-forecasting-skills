#!/usr/bin/env python3
"""Deterministic first-pass audit for forecasting datasets."""

from __future__ import annotations

import argparse
import json
import math
import numbers
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - exercised only when dependency is absent.
    raise SystemExit(
        "forecasting_data_audit.py requires pandas. Install it in your environment, "
        "for example: python -m pip install pandas"
    ) from exc


TIME_NAME_HINTS = (
    "ds",
    "date",
    "datetime",
    "timestamp",
    "time",
    "period",
    "week",
    "month",
)
TARGET_NAME_HINTS = (
    "y",
    "target",
    "value",
    "sales",
    "demand",
    "qty",
    "quantity",
    "count",
    "revenue",
    "load",
    "volume",
)
ID_NAME_HINTS = (
    "unique_id",
    "series_id",
    "item_id",
    "store_id",
    "sku",
    "location",
    "meter_id",
    "id",
)
KNOWN_FUTURE_HINTS = (
    "holiday",
    "calendar",
    "dayofweek",
    "day_of_week",
    "dow",
    "month",
    "week",
    "quarter",
    "year",
    "promo",
    "promotion",
    "event",
    "planned",
    "schedule",
)


def read_frame(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        return pd.read_json(path)
    raise SystemExit(f"Unsupported file type: {suffix}. Use csv, parquet, json, or jsonl.")


def parse_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def clean_number(value: Any) -> Any:
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return str(value)
    if isinstance(value, dict):
        return {str(k): clean_number(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_number(v) for v in value]
    return value


def name_score(name: str, hints: tuple[str, ...]) -> int:
    normalized = name.lower().strip()
    return max((10 if normalized == hint else 5 for hint in hints if hint in normalized), default=0)


def datetime_parse_rate(series: pd.Series, sample_size: int = 1000) -> float:
    sample = series.dropna().head(sample_size)
    if sample.empty:
        return 0.0
    parsed = pd.to_datetime(sample, errors="coerce")
    return float(parsed.notna().mean())


def infer_time_col(df: pd.DataFrame, explicit: str | None) -> tuple[str | None, list[dict[str, Any]]]:
    if explicit:
        return explicit, []
    candidates: list[dict[str, Any]] = []
    for col in df.columns:
        dtype = df[col].dtype
        hint_score = name_score(str(col), TIME_NAME_HINTS)
        if pd.api.types.is_datetime64_any_dtype(dtype):
            parse_rate = 1.0
        elif pd.api.types.is_numeric_dtype(dtype) and hint_score == 0:
            parse_rate = 0.0
        else:
            parse_rate = datetime_parse_rate(df[col])
        score = hint_score + int(parse_rate * 10)
        if score >= 8:
            candidates.append({"column": str(col), "score": score, "parse_rate": parse_rate})
    candidates.sort(key=lambda item: (-item["score"], item["column"]))
    return (candidates[0]["column"] if candidates else None), candidates[:5]


def infer_target_col(
    df: pd.DataFrame, explicit: str | None, excluded: set[str]
) -> tuple[str | None, list[dict[str, Any]]]:
    if explicit:
        return explicit, []
    candidates: list[dict[str, Any]] = []
    for col in df.columns:
        if col in excluded:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        non_null_rate = float(df[col].notna().mean())
        unique_rate = float(df[col].nunique(dropna=True) / max(len(df), 1))
        score = name_score(str(col), TARGET_NAME_HINTS) + int(non_null_rate * 5) + int(unique_rate * 3)
        candidates.append(
            {
                "column": str(col),
                "score": score,
                "non_null_rate": non_null_rate,
                "unique_rate": unique_rate,
            }
        )
    candidates.sort(key=lambda item: (-item["score"], item["column"]))
    return (candidates[0]["column"] if candidates else None), candidates[:5]


def infer_id_cols(df: pd.DataFrame, explicit: list[str], excluded: set[str]) -> tuple[list[str], list[str]]:
    if explicit:
        return explicit, []
    candidates: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if name_score(str(col), ID_NAME_HINTS) <= 0:
            continue
        nunique = df[col].nunique(dropna=True)
        if 1 < nunique < len(df):
            candidates.append(str(col))
    if "unique_id" in candidates:
        return ["unique_id"], candidates
    return [], candidates


def parse_time(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    raw = series.dropna().astype(str)
    tz_tokens = raw.str.contains(r"(?:Z$|[+-]\d{2}:?\d{2}$|\bUTC\b|\bGMT\b)", regex=True).mean()
    use_utc = bool(tz_tokens > 0)
    parsed = pd.to_datetime(series, errors="coerce", utc=use_utc)
    metadata = {
        "timezone_status": "timezone_tokens_present_converted_to_utc" if use_utc else "naive_or_not_encoded",
        "timezone_token_rate": float(tz_tokens) if not raw.empty else 0.0,
        "parse_nulls": int(parsed.isna().sum()),
        "parse_null_rate": float(parsed.isna().mean()) if len(parsed) else 0.0,
    }
    return parsed, metadata


def group_iterator(df: pd.DataFrame, id_cols: list[str]):
    if not id_cols:
        yield "__single_series__", df
        return
    group_key: str | list[str] = id_cols[0] if len(id_cols) == 1 else id_cols
    for key, group in df.groupby(group_key, dropna=False, sort=False):
        if isinstance(key, tuple):
            name = "|".join(str(part) for part in key)
        else:
            name = str(key)
        yield name, group


def infer_frequency(df: pd.DataFrame, time_col: str, id_cols: list[str]) -> dict[str, Any]:
    per_series: list[dict[str, Any]] = []
    all_deltas: list[pd.Timedelta] = []
    for name, group in group_iterator(df, id_cols):
        times = group[time_col].dropna().drop_duplicates().sort_values()
        inferred = None
        if len(times) >= 3:
            try:
                inferred = pd.infer_freq(times)
            except ValueError:
                inferred = None
        deltas = times.diff().dropna()
        modal_delta = None
        if not deltas.empty:
            counts = deltas.value_counts()
            modal_delta = counts.index[0]
            all_deltas.extend(deltas.tolist())
        per_series.append(
            {
                "series": name,
                "observations": int(len(group)),
                "unique_timestamps": int(len(times)),
                "start": times.min().isoformat() if len(times) else None,
                "end": times.max().isoformat() if len(times) else None,
                "inferred_freq": inferred,
                "modal_delta": str(modal_delta) if modal_delta is not None else None,
            }
        )
    global_modal_delta = None
    if all_deltas:
        global_modal_delta = pd.Series(all_deltas).value_counts().index[0]
    inferred_values = [item["inferred_freq"] for item in per_series if item["inferred_freq"]]
    global_inferred = None
    if inferred_values:
        global_inferred = pd.Series(inferred_values).value_counts().index[0]
    return {
        "global_inferred_freq": global_inferred,
        "global_modal_delta": str(global_modal_delta) if global_modal_delta is not None else None,
        "per_series_sample": per_series[:20],
        "series_with_inferred_freq": int(len(inferred_values)),
    }


def duplicate_count(df: pd.DataFrame, key_cols: list[str]) -> int:
    if not key_cols:
        return 0
    return int(df.duplicated(key_cols, keep=False).sum())


def gap_summary(df: pd.DataFrame, time_col: str, id_cols: list[str], freq: str | None) -> dict[str, Any]:
    if not freq:
        return {"checked": False, "reason": "No explicit or inferred regular frequency available."}
    total_missing = 0
    series_with_gaps = 0
    examples: list[dict[str, Any]] = []
    for name, group in group_iterator(df, id_cols):
        times = group[time_col].dropna().drop_duplicates().sort_values()
        if len(times) < 2:
            continue
        try:
            expected = pd.date_range(times.iloc[0], times.iloc[-1], freq=freq)
        except ValueError as exc:
            return {"checked": False, "reason": f"Could not build expected range with freq={freq}: {exc}"}
        missing = expected.difference(pd.DatetimeIndex(times))
        missing_count = int(len(missing))
        total_missing += missing_count
        if missing_count:
            series_with_gaps += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "series": name,
                        "missing_count": missing_count,
                        "first_missing": missing[0].isoformat(),
                        "last_missing": missing[-1].isoformat(),
                    }
                )
    return {
        "checked": True,
        "freq": freq,
        "missing_timestamps": total_missing,
        "series_with_gaps": series_with_gaps,
        "examples": examples,
    }


def missing_summary(df: pd.DataFrame) -> dict[str, Any]:
    missing = df.isna().sum().sort_values(ascending=False)
    rows = []
    for col, count in missing.head(20).items():
        if int(count) == 0:
            continue
        rows.append({"column": str(col), "missing": int(count), "missing_rate": float(count / max(len(df), 1))})
    return {"top_columns": rows}


def outlier_summary(df: pd.DataFrame, target_col: str, id_cols: list[str]) -> dict[str, Any]:
    if target_col not in df.columns or not pd.api.types.is_numeric_dtype(df[target_col]):
        return {"checked": False, "reason": "Target is missing or non-numeric."}
    total = 0
    examples: list[dict[str, Any]] = []
    for name, group in group_iterator(df, id_cols):
        y = group[target_col].dropna()
        if len(y) < 8:
            continue
        q1 = y.quantile(0.25)
        q3 = y.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0 or pd.isna(iqr):
            continue
        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        count = int(((y < low) | (y > high)).sum())
        total += count
        if count and len(examples) < 10:
            examples.append({"series": name, "outliers": count, "lower_bound": float(low), "upper_bound": float(high)})
    return {"checked": True, "method": "per-series IQR 1.5x", "outlier_rows": total, "examples": examples}


def exogenous_summary(df: pd.DataFrame, excluded: set[str]) -> dict[str, Any]:
    candidates = [str(col) for col in df.columns if col not in excluded]
    known_future_like = [col for col in candidates if name_score(col, KNOWN_FUTURE_HINTS) > 0]
    high_missing = [col for col in candidates if df[col].isna().mean() >= 0.2]
    return {
        "candidate_columns": candidates,
        "known_future_name_candidates": known_future_like,
        "high_missing_candidate_columns": high_missing,
        "requires_classification": candidates,
    }


def split_readiness(
    df: pd.DataFrame,
    time_col: str,
    id_cols: list[str],
    horizon: int | None,
    gap: int,
) -> dict[str, Any]:
    if not horizon:
        return {"checked": False, "reason": "Pass --horizon to check split readiness."}
    short_series = []
    min_required = horizon * 3 + gap
    for name, group in group_iterator(df, id_cols):
        n_obs = int(group[time_col].dropna().nunique())
        if n_obs < min_required:
            short_series.append({"series": name, "unique_timestamps": n_obs, "minimum_recommended": min_required})
    return {
        "checked": True,
        "horizon": horizon,
        "gap": gap,
        "minimum_recommended_unique_timestamps": min_required,
        "short_series_count": len(short_series),
        "short_series_examples": short_series[:20],
        "recommendation": "Use temporal train/validation/test cutoffs or rolling-origin backtesting; never random split.",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    path = Path(args.path)
    df = read_frame(path)
    report: dict[str, Any] = {
        "input": {"path": str(path), "rows": int(len(df)), "columns": int(len(df.columns))},
        "schema": {str(col): str(dtype) for col, dtype in df.dtypes.items()},
        "errors": [],
        "warnings": [],
    }

    time_col, time_candidates = infer_time_col(df, args.time_col)
    if not time_col or time_col not in df.columns:
        report["errors"].append("Could not identify a valid time column. Pass --time-col.")
        report["inferred_columns"] = {"time_col": time_col, "time_candidates": time_candidates}
        return report

    parsed_time, time_metadata = parse_time(df[time_col])
    work = df.copy()
    work[time_col] = parsed_time
    excluded_for_target = {time_col}
    target_col, target_candidates = infer_target_col(work, args.target_col, excluded_for_target)
    if not target_col or target_col not in work.columns:
        report["errors"].append("Could not identify a valid target column. Pass --target-col.")
        report["inferred_columns"] = {
            "time_col": time_col,
            "time_candidates": time_candidates,
            "target_col": target_col,
            "target_candidates": target_candidates,
        }
        return report

    id_cols, id_candidates = infer_id_cols(work, parse_list(args.id_cols), {time_col, target_col})
    invalid_id_cols = [col for col in id_cols if col not in work.columns]
    if invalid_id_cols:
        report["errors"].append(f"ID columns not found: {invalid_id_cols}")
        id_cols = [col for col in id_cols if col in work.columns]

    sort_cols = id_cols + [time_col]
    work = work.sort_values(sort_cols)
    key_cols = id_cols + [time_col]
    freq_report = infer_frequency(work, time_col, id_cols)
    gap_freq = args.freq or freq_report.get("global_inferred_freq")

    report["inferred_columns"] = {
        "time_col": time_col,
        "time_candidates": time_candidates,
        "target_col": target_col,
        "target_candidates": target_candidates,
        "id_cols": id_cols,
        "id_candidates": id_candidates,
    }
    report["time"] = {
        **time_metadata,
        "min": work[time_col].min().isoformat() if work[time_col].notna().any() else None,
        "max": work[time_col].max().isoformat() if work[time_col].notna().any() else None,
        "globally_monotonic_after_sort": bool(work[time_col].is_monotonic_increasing) if not id_cols else None,
    }
    report["panel"] = {
        "id_cols": id_cols,
        "n_series": int(work.groupby(id_cols, dropna=False).ngroups) if id_cols else 1,
    }
    report["frequency"] = freq_report
    report["quality"] = {
        "duplicate_key_rows": duplicate_count(work, key_cols),
        "target_missing": int(work[target_col].isna().sum()),
        "target_missing_rate": float(work[target_col].isna().mean()),
        "missing_values": missing_summary(work),
        "gaps": gap_summary(work, time_col, id_cols, gap_freq),
        "outliers": outlier_summary(work, target_col, id_cols),
    }
    report["exogenous"] = exogenous_summary(work, set(key_cols + [target_col]))
    report["split_readiness"] = split_readiness(work, time_col, id_cols, args.horizon, args.gap)

    if report["quality"]["duplicate_key_rows"]:
        report["warnings"].append("Duplicate forecasting keys found; resolve before splitting or modeling.")
    if report["quality"]["target_missing"]:
        report["warnings"].append("Missing target values found; choose a train-only imputation/drop/flag strategy.")
    gaps = report["quality"]["gaps"]
    if gaps.get("checked") and gaps.get("missing_timestamps", 0):
        report["warnings"].append("Temporal gaps found; decide whether to resample, impute, or use a model that supports irregular data.")
    if report["exogenous"]["requires_classification"]:
        report["warnings"].append("Classify every exogenous candidate by future availability before modeling.")
    if target_candidates and len(target_candidates) > 1 and not args.target_col:
        report["warnings"].append("Target column was inferred; verify it explicitly before modeling.")
    if time_candidates and len(time_candidates) > 1 and not args.time_col:
        report["warnings"].append("Time column was inferred; verify it explicitly before modeling.")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a tabular dataset before forecasting.")
    parser.add_argument("path", help="Input CSV, Parquet, JSON, or JSONL file.")
    parser.add_argument("--time-col", help="Timestamp column name.")
    parser.add_argument("--target-col", help="Target column name.")
    parser.add_argument("--id-cols", help="Comma-separated series identifier columns.")
    parser.add_argument("--freq", help="Expected pandas frequency alias, for example D, h, W, MS.")
    parser.add_argument("--horizon", type=int, help="Forecast horizon in periods.")
    parser.add_argument("--gap", type=int, default=0, help="Gap periods between train end and forecast start.")
    args = parser.parse_args()
    report = build_report(args)
    print(json.dumps(clean_number(report), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
