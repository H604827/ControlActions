#!/usr/bin/env python3
"""
Plant-context feature builder — single point-in-time.

Given a PV/OP time-series parquet, an operating-limits CSV, a query timestamp and
(optionally) a deviation-start timestamp, this computes the plant-context feature
vector used for control-action similarity retrieval.

It is the runtime counterpart of the per-merged-action table built in
EXPERIMENTS/control_actions_context.ipynb: the feature definitions and naming
(<tag>_<suffix>) match, so a vector produced here can be compared directly against
the historical index.

Context features (the similarity key), for the alarm tag and every related PV tag:
  - pv_now            raw PV at the query time
  - norm_pos          (pv_now - lower) / range          -> position in operating band
  - pv_dev_start      raw PV at the deviation-start time
  - dev_norm_pos      (pv_dev - lower) / range
  - episode_change    (pv_now - pv_dev) / range          -> drift since deviation
  - roc_st/mt/lt      (pv_now - pv_{t-W}) / range         -> look-back ROC (3/10/30 min)
  - roc_dir_st/mt/lt  increasing / decreasing / same      (deadband on normalized ROC)
  - traj_vs_limit     towards_limit / away_from_limit / stable (vs nearest limit)
Alarm-tag only:
  - alarm_proximity        (pv_now - alarm_thr) / (upper - alarm_thr)
  - minutes_since_deviation

All ROC/position features are normalized by the tag's operating range, so tags with
different engineering units are comparable, and nothing uses look-ahead information.

Usage (CLI):
    python SRC/plant_context_features.py \
        --pv-file DATA/PV-OP_data/03LIC_1071_JAN_2026.parquet \
        --query-time "2025-01-06 16:30:00" \
        --deviation-start "2025-01-06 16:10:00" \
        --output /tmp/context_vector.json --long

Usage (import):
    from SRC.plant_context_features import (
        load_pv_data, load_operating_limits, ContextConfig, build_context_vector,
    )
    pv = load_pv_data("DATA/PV-OP_data/03LIC_1071_JAN_2026.parquet")
    lim = load_operating_limits("DATA/operating_limits.csv")
    vec = build_context_vector(pv, lim, query_time="2025-01-06 16:30:00",
                               deviation_start="2025-01-06 16:10:00")
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

DEFAULT_LIMITS_PATH = "DATA/operating_limits.csv"
DEFAULT_TARGET_TAG = "03LIC_1071.PV"
DEFAULT_ALARM_THRESHOLD = 28.75
DEFAULT_WINDOWS = {"st": 3, "mt": 10, "lt": 30}   # look-back minutes: short / medium / long
DEFAULT_DEADBAND = 0.002                          # |normalized ROC| below this -> "same" / "stable"


@dataclass
class ContextConfig:
    """Configuration for the plant-context feature builder."""
    target_tag: str = DEFAULT_TARGET_TAG
    alarm_threshold: float = DEFAULT_ALARM_THRESHOLD
    windows: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_WINDOWS))
    deadband: float = DEFAULT_DEADBAND


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_pv_data(path: str | Path) -> pd.DataFrame:
    """Load a PV/OP parquet and return it with a sorted DatetimeIndex.

    Accepts either a ``TimeStamp`` column or an already-datetime index.
    """
    df = pd.read_parquet(path)
    if "TimeStamp" in df.columns:
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"])
        df = df.set_index("TimeStamp")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


def load_operating_limits(path: str | Path = DEFAULT_LIMITS_PATH) -> pd.DataFrame:
    """Load operating limits for ``.PV`` tags, indexed by tag name, with a ``range`` column."""
    limits = pd.read_csv(path)
    limits = limits[limits["TAG_NAME"].astype(str).str.endswith(".PV")].copy()
    limits["range"] = limits["UPPER_LIMIT"] - limits["LOWER_LIMIT"]
    limits = limits[limits["range"] > 0].drop_duplicates(subset=["TAG_NAME"])
    limits = limits.set_index("TAG_NAME")
    return limits


# ---------------------------------------------------------------------------
# As-of snapshot at a single timestamp
# ---------------------------------------------------------------------------
def asof_snapshot_point(frame: pd.DataFrame, timestamp) -> pd.Series:
    """Last known values of every column at or before ``timestamp`` (ffill).

    Falls back to the earliest available row (bfill) when ``timestamp`` precedes the
    data, matching the notebook's ``asof_snapshot`` behaviour. Returns an all-NaN
    Series when the frame is empty.
    """
    if timestamp is None or (isinstance(timestamp, float) and math.isnan(timestamp)) or pd.isna(timestamp):
        return pd.Series(np.nan, index=frame.columns)
    ts = pd.Timestamp(timestamp)
    idx = frame.index.get_indexer([ts], method="ffill")[0]
    if idx == -1:
        idx = frame.index.get_indexer([ts], method="bfill")[0]
    if idx == -1:
        return pd.Series(np.nan, index=frame.columns)
    return frame.iloc[idx]


# ---------------------------------------------------------------------------
# Classification helpers (operate on a per-tag Series)
# ---------------------------------------------------------------------------
def _classify_roc_direction(norm_roc: pd.Series, deadband: float) -> pd.Series:
    """increasing / decreasing / same on a normalized-ROC series (NaN -> NaN)."""
    labels = np.where(
        norm_roc.to_numpy() > deadband, "increasing",
        np.where(norm_roc.to_numpy() < -deadband, "decreasing", "same"),
    )
    out = pd.Series(labels, index=norm_roc.index, dtype=object)
    out[norm_roc.isna()] = np.nan
    return out


def _classify_trajectory(norm_pos: pd.Series, norm_roc: pd.Series, deadband: float) -> pd.Series:
    """towards_limit / away_from_limit / stable relative to the NEAREST limit.

    Nearest limit is chosen by normalized position: <= 0.5 -> lower bound, else upper.
    Returns NaN where either input is NaN.
    """
    pos = norm_pos.to_numpy()
    roc = norm_roc.to_numpy()
    nearest_lower = pos <= 0.5
    towards = np.where(nearest_lower, roc < -deadband, roc > deadband)
    labels = np.where(
        np.abs(roc) <= deadband, "stable",
        np.where(towards, "towards_limit", "away_from_limit"),
    )
    out = pd.Series(labels, index=norm_pos.index, dtype=object)
    out[norm_pos.isna() | norm_roc.isna()] = np.nan
    return out


# ---------------------------------------------------------------------------
# Core: per-tag context table at a single point
# ---------------------------------------------------------------------------
def compute_context_table(
    pv_df: pd.DataFrame,
    limits_df: pd.DataFrame,
    query_time,
    deviation_start=None,
    config: Optional[ContextConfig] = None,
    related_tags: Optional[list] = None,
) -> pd.DataFrame:
    """Per-tag context features at ``query_time`` (one row per PV tag).

    Args:
        pv_df: PV/OP time series with a sorted DatetimeIndex.
        limits_df: operating limits (see ``load_operating_limits``).
        query_time: the "particular point" at which to evaluate features.
        deviation_start: excursion start (from upstream anomaly detection). When
            None, deviation-anchored features are NaN.
        config: ContextConfig; defaults used when None.
        related_tags: optional explicit tag list; defaults to every ``.PV`` tag in
            ``pv_df`` that also has valid operating limits.

    Returns:
        DataFrame indexed by tag name with the per-tag context feature columns.
    """
    config = config or ContextConfig()

    pv_tags = [c for c in pv_df.columns if c.endswith(".PV")]
    valid = [t for t in pv_tags if t in limits_df.index]
    if related_tags is not None:
        wanted = set(related_tags)
        valid = [t for t in valid if t in wanted or t.replace(".PV", "") in wanted]
    if config.target_tag in limits_df.index and config.target_tag in pv_df.columns \
            and config.target_tag not in valid:
        valid.append(config.target_tag)
    if not valid:
        raise ValueError("No PV tags with valid operating limits were found in the data.")

    lower = limits_df.loc[valid, "LOWER_LIMIT"]
    rng = limits_df.loc[valid, "range"]

    cur = asof_snapshot_point(pv_df[valid], query_time)
    dev = asof_snapshot_point(pv_df[valid], deviation_start)
    window_snaps = {
        name: asof_snapshot_point(pv_df[valid], pd.Timestamp(query_time) - pd.Timedelta(minutes=minutes))
        for name, minutes in config.windows.items()
    }

    table = pd.DataFrame(index=pd.Index(valid, name="tag"))
    table["pv_now"] = cur
    table["norm_pos"] = (cur - lower) / rng
    table["pv_dev_start"] = dev
    table["dev_norm_pos"] = (dev - lower) / rng
    table["episode_change"] = (cur - dev) / rng
    for name in config.windows:
        table[f"roc_{name}"] = (cur - window_snaps[name]) / rng
    for name in config.windows:
        table[f"roc_dir_{name}"] = _classify_roc_direction(table[f"roc_{name}"], config.deadband)
    # short-term movement drives the trajectory-vs-limit label
    st_name = "st" if "st" in config.windows else next(iter(config.windows))
    table["traj_vs_limit"] = _classify_trajectory(table["norm_pos"], table[f"roc_{st_name}"], config.deadband)
    return table


def build_context_vector(
    pv_df: pd.DataFrame,
    limits_df: pd.DataFrame,
    query_time,
    deviation_start=None,
    config: Optional[ContextConfig] = None,
    related_tags: Optional[list] = None,
) -> Dict[str, object]:
    """Flat context feature vector (the similarity key) at ``query_time``.

    Keys are ``<tag_stub>_<suffix>`` (tag_stub drops the ``.PV`` suffix), matching the
    notebook's ``pm_features_df`` columns, plus the alarm-tag extras
    ``<target>_alarm_proximity`` and ``minutes_since_deviation``.
    """
    config = config or ContextConfig()
    table = compute_context_table(pv_df, limits_df, query_time, deviation_start, config, related_tags)

    vector: Dict[str, object] = {}
    for tag, row in table.iterrows():
        stub = tag.replace(".PV", "")
        for col, value in row.items():
            vector[f"{stub}_{col}"] = value

    # alarm-tag-only extras
    target = config.target_tag
    target_stub = target.replace(".PV", "")
    if target in table.index and target in limits_df.index:
        target_upper = limits_df.loc[target, "UPPER_LIMIT"]
        pv_now_target = table.loc[target, "pv_now"]
        denom = target_upper - config.alarm_threshold
        vector[f"{target_stub}_alarm_proximity"] = (
            (pv_now_target - config.alarm_threshold) / denom if denom else np.nan
        )

    if deviation_start is not None and not pd.isna(deviation_start):
        minutes = (pd.Timestamp(query_time) - pd.Timestamp(deviation_start)).total_seconds() / 60.0
    else:
        minutes = np.nan
    vector["minutes_since_deviation"] = minutes
    return vector


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _json_safe(obj):
    """Recursively convert numpy scalars and NaN to JSON-serialisable values."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if (obj != obj) else float(obj)  # NaN -> None
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def _parse_windows(text: str) -> Dict[str, int]:
    """Parse '3,10,30' (assigned st/mt/lt) or 'st=3,mt=10,lt=30' into a dict."""
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if all("=" in p for p in parts):
        return {k.strip(): int(v) for k, v in (p.split("=") for p in parts)}
    names = ["st", "mt", "lt"]
    values = [int(p) for p in parts]
    return {names[i] if i < len(names) else f"w{i}": v for i, v in enumerate(values)}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute plant-context (PM dimension) features at a single point in time."
    )
    parser.add_argument("--pv-file", required=True, help="Path to the PV/OP parquet file")
    parser.add_argument("--limits-file", default=DEFAULT_LIMITS_PATH, help="Operating limits CSV")
    parser.add_argument("--query-time", required=True, help="Timestamp to evaluate features at (ISO)")
    parser.add_argument("--deviation-start", default=None,
                        help="Deviation-start timestamp (ISO). Omit if unknown.")
    parser.add_argument("--target-tag", default=DEFAULT_TARGET_TAG, help="Alarm/target PV tag")
    parser.add_argument("--alarm-threshold", type=float, default=DEFAULT_ALARM_THRESHOLD,
                        help="Alarm threshold for the target tag")
    parser.add_argument("--windows", default="3,10,30",
                        help="Look-back windows in minutes, e.g. '3,10,30' or 'st=3,mt=10,lt=30'")
    parser.add_argument("--deadband", type=float, default=DEFAULT_DEADBAND,
                        help="Normalized-ROC deadband for direction/trajectory labels")
    parser.add_argument("--related-tags", default=None,
                        help="Optional comma-separated tag list to restrict context tags")
    parser.add_argument("--output", default=None, help="Write the vector to this .json or .csv path")
    parser.add_argument("--long", action="store_true", help="Also print the per-tag context table")
    return parser.parse_args()


def main():
    args = parse_args()

    config = ContextConfig(
        target_tag=args.target_tag,
        alarm_threshold=args.alarm_threshold,
        windows=_parse_windows(args.windows),
        deadband=args.deadband,
    )
    related = None
    if args.related_tags:
        related = [t.strip() for t in args.related_tags.split(",") if t.strip()]

    print(f"Loading PV data     : {args.pv_file}")
    pv_df = load_pv_data(args.pv_file)
    print(f"  rows={len(pv_df):,}  range={pv_df.index.min()} .. {pv_df.index.max()}")
    limits_df = load_operating_limits(args.limits_file)
    print(f"Operating limits    : {args.limits_file}  ({len(limits_df)} PV tags)")
    print(f"Query time          : {args.query_time}")
    print(f"Deviation start     : {args.deviation_start if args.deviation_start else '(none - deviation features NaN)'}")
    print(f"Windows (min)       : {config.windows}")

    table = compute_context_table(pv_df, limits_df, args.query_time, args.deviation_start, config, related)
    vector = build_context_vector(pv_df, limits_df, args.query_time, args.deviation_start, config, related)

    target_stub = config.target_tag.replace(".PV", "")
    print(f"\nContext tags        : {len(table)}  (target present: {config.target_tag in table.index})")
    print(f"Context features    : {len(vector)}")
    print(f"\nAlarm tag ({config.target_tag}) summary:")
    for suffix in ["pv_now", "norm_pos", "pv_dev_start", "episode_change",
                   "roc_st", "roc_mt", "roc_lt", "roc_dir_st", "traj_vs_limit"]:
        key = f"{target_stub}_{suffix}"
        if key in vector:
            print(f"  {key:<40} {vector[key]}")
    print(f"  {target_stub + '_alarm_proximity':<40} {vector.get(target_stub + '_alarm_proximity')}")
    print(f"  {'minutes_since_deviation':<40} {vector.get('minutes_since_deviation')}")

    if args.long:
        print("\nPer-tag context table:")
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(table.round(4).to_string())

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".csv":
            table.to_csv(out_path)
        else:
            with open(out_path, "w") as fh:
                json.dump(_json_safe(vector), fh, indent=2)
        print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
