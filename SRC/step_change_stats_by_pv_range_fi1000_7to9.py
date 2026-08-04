"""
Step-change statistics of 03LIC_1071-alarm control actions, restricted to
alarms where the feed rate (02FI_1000.PV) averaged between 7 and 9 over a
tight +/-30 minute window. EVERY tag's OP/SP step changes are bucketed by
03LIC_1071.PV (the alarm tag's PV) at the time of the action.

  1. Alarm (cluster) universe restricted to those where the MEAN 02FI_1000.PV
     over [cluster_start - 30min, cluster_end + 30min] falls within [7, 9].
  2. Control actions considered are restricted to that SAME +/-30 minute
     window (re-derived directly from alarm_clusters, not assumed from the
     'before'/'during'/'after' labels already in the workbook).
  3. The report lists EVERY tag operated (OP/SP) in that window -- no hardcoded
     tag list -- sorted most-operated first, and breaks each one out per tag
     with its step changes bucketed by 03LIC_1071.PV, plus a Value_Range
     column (min-max of PrevValue/Value) per PV bucket. A tag that has no
     .PV/.OP column of its own is still included (bucketing uses 1071's PV).

Output: a Markdown report at RESULTS/step_change_analysis_fi1000_7to9/
        step_change_analysis_fi1000_7to9.md
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
WORKBOOK = REPO / "DATA/1071_pvlo_alarms_clustered_with_control_actions_with_plant_context.xlsx"
PV_PARQUET = REPO / "DATA/PV-OP_data/03LIC_1071_JAN_2026.parquet"

FI1000_COL = "02FI_1000.PV"
TARGET_PV_COL = "03LIC_1071.PV"   # the alarm tag; ALL step changes are bucketed by this PV

WINDOW_BEFORE = pd.Timedelta(minutes=30)
WINDOW_AFTER = pd.Timedelta(minutes=30)
FI1000_LOWER = 7.0
FI1000_UPPER = 9.0

N_BINS = 10
ASOF_TOLERANCE = pd.Timedelta("5min")

OUT_DIR = REPO / "RESULTS/step_change_analysis_fi1000_7to9"
OUT_MD = OUT_DIR / "step_change_analysis_fi1000_7to9.md"


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_clusters() -> pd.DataFrame:
    """Unique alarm clusters (== '1071 alarms') with their +/-30min window."""
    ac = pd.read_excel(
        WORKBOOK, sheet_name="alarm_clusters",
        usecols=["cluster_id", "cluster_start_time", "cluster_end_time"],
    )
    ac["cluster_start_time"] = pd.to_datetime(ac["cluster_start_time"])
    ac["cluster_end_time"] = pd.to_datetime(ac["cluster_end_time"])
    clusters = (
        ac.drop_duplicates(subset=["cluster_id"])
        .dropna()
        .sort_values("cluster_id")
        .reset_index(drop=True)
    )
    clusters["window_start"] = clusters["cluster_start_time"] - WINDOW_BEFORE
    clusters["window_end"] = clusters["cluster_end_time"] + WINDOW_AFTER
    return clusters


def load_pv_series(col: str) -> pd.Series:
    pv = pd.read_parquet(PV_PARQUET, columns=["TimeStamp", col])
    pv["TimeStamp"] = pd.to_datetime(pv["TimeStamp"])
    pv = pv.dropna(subset=["TimeStamp", col]).sort_values("TimeStamp")
    series = pv.set_index("TimeStamp")[col]
    return series[~series.index.duplicated(keep="last")]


def compute_fi1000_stats(clusters: pd.DataFrame, fi1000_series: pd.Series) -> pd.DataFrame:
    """Min/max of 02FI_1000.PV within each cluster's +/-30min window.

    A cluster later qualifies only if EVERY sample in this window is within
    [FI1000_LOWER, FI1000_UPPER] (i.e. window min >= lower AND max <= upper).
    """
    mins, maxs, n_samples = [], [], []
    for _, row in clusters.iterrows():
        window = fi1000_series.loc[row["window_start"]: row["window_end"]].dropna()
        if window.empty:
            mins.append(np.nan)
            maxs.append(np.nan)
            n_samples.append(0)
        else:
            mins.append(float(window.min()))
            maxs.append(float(window.max()))
            n_samples.append(len(window))
    clusters = clusters.copy()
    clusters["fi1000_min"] = mins
    clusters["fi1000_max"] = maxs
    clusters["fi1000_n_samples"] = n_samples
    return clusters


# --------------------------------------------------------------------------- #
# Step-change analysis (per tag, per action type)
# --------------------------------------------------------------------------- #
def make_pv_bins(pv: pd.Series, n_bins: int = N_BINS) -> pd.Series:
    """Quantile-bin a PV series by 03LIC_1071.PV. Robust to few/identical values:
    every input value is always assigned to an Interval bin (never NaN)."""
    pv = pd.Series(pv).astype(float)
    n_unique = pv.nunique()
    if n_unique <= 1:
        v = float(pv.iloc[0])
        eps = max(abs(v) * 1e-3, 1e-3)
        return pd.cut(pv, bins=[v - eps, v + eps])
    bins = min(n_bins, n_unique)
    while bins >= 1:
        try:
            binned = pd.qcut(pv, q=bins, duplicates="drop")
        except ValueError:
            binned = None
        if binned is not None and binned.notna().all():
            return binned
        bins -= 1
    # Fall back to a single equal-width bin spanning the whole range.
    return pd.cut(pv, bins=1)


def format_interval(iv, nd: int) -> str:
    return f"{iv.left:.{nd}f} - {iv.right:.{nd}f}"


def required_precision(intervals, max_nd: int = 6) -> int:
    """Smallest decimal precision that keeps all interval labels distinct AND
    keeps each individual bin's own left/right edges from rounding to the
    same text (which would render as a confusing "X - X")."""
    uniq = [iv for iv in dict.fromkeys(intervals) if isinstance(iv, pd.Interval)]
    if not uniq:
        return 1
    for nd in range(1, max_nd + 1):
        labels = [format_interval(iv, nd) for iv in uniq]
        no_collapsed_bin = all(f"{iv.left:.{nd}f}" != f"{iv.right:.{nd}f}" for iv in uniq)
        if len(set(labels)) == len(labels) and no_collapsed_bin:
            return nd
    return max_nd


def analyze_tag(actions: pd.DataFrame, tag: str, action_type: str):
    """Return (step_stats_df, directional_df) for one Source tag + Description.

    Actions are bucketed by 03LIC_1071.PV (the alarm tag's PV) at the action
    time, which is pre-attached to `actions` as the `target_pv` column.
    """
    sub = actions[(actions["Source"] == tag) & (actions["Description"] == action_type)].copy()
    if sub.empty:
        return None, None, 0, 0

    raw_count = len(sub)
    sub["PrevValue_num"] = pd.to_numeric(sub["PrevValue"], errors="coerce")
    sub["Value_num"] = pd.to_numeric(sub["Value"], errors="coerce")
    sub = sub.dropna(subset=["PrevValue_num", "Value_num", "target_pv"])
    numeric_count = len(sub)
    if sub.empty:
        return None, None, raw_count, numeric_count

    sub["step_change"] = sub["Value_num"] - sub["PrevValue_num"]
    sub["PV_Range"] = make_pv_bins(sub["target_pv"])
    label_nd = required_precision(sub["PV_Range"].tolist())

    def value_range(g):
        vals = pd.concat([g["PrevValue_num"], g["Value_num"]])
        return f"{vals.min():.2f} to {vals.max():.2f}"

    rows = []
    for pv_range, g in sub.groupby("PV_Range", observed=True):
        rows.append({
            "PV_Range": pv_range,
            "Data_Points": len(g),
            "Step_Min": g["step_change"].min(),
            "Step_Max": g["step_change"].max(),
            "Step_Mean": g["step_change"].mean(),
            "Step_Median": g["step_change"].median(),
            "Step_Std": g["step_change"].std(),
            "Step_25th": g["step_change"].quantile(0.25),
            "Step_75th": g["step_change"].quantile(0.75),
            "Value_Range": value_range(g),
        })
    step_stats = pd.DataFrame(rows).sort_values("PV_Range").reset_index(drop=True)
    step_stats["PV_Range"] = step_stats["PV_Range"].apply(lambda iv: format_interval(iv, label_nd))

    sub["Step_Direction"] = np.where(
        sub["step_change"] > 0, "Positive", np.where(sub["step_change"] < 0, "Negative", "Zero")
    )
    dir_rows = []
    for pv_range, g in sub.groupby("PV_Range", observed=True):
        total = len(g)
        pos = (g["Step_Direction"] == "Positive").sum()
        neg = (g["Step_Direction"] == "Negative").sum()
        zero = (g["Step_Direction"] == "Zero").sum()
        dir_rows.append({
            "PV_Range": pv_range,
            "Data_Points": total,
            "Positive_Steps": pos,
            "Negative_Steps": neg,
            "Zero_Steps": zero,
            "Pos_Steps_%": round(100 * pos / total, 2),
            "Neg_Steps_%": round(100 * neg / total, 2),
        })
    directional = pd.DataFrame(dir_rows).sort_values("PV_Range").reset_index(drop=True)
    directional["PV_Range"] = directional["PV_Range"].apply(lambda iv: format_interval(iv, label_nd))

    return step_stats, directional, raw_count, numeric_count


# --------------------------------------------------------------------------- #
# Markdown helpers
# --------------------------------------------------------------------------- #
def fmt_num(x, nd=3):
    if pd.isna(x):
        return "NaN"
    return f"{x:.{nd}f}"


def df_to_md(df: pd.DataFrame, float_cols=(), nd=3) -> str:
    display = df.copy()
    for c in float_cols:
        if c in display.columns:
            display[c] = display[c].apply(lambda x: fmt_num(x, nd))
    cols = list(display.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in display.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    md = []
    md.append("# Step Change Analysis — 03LIC_1071 (FI1000 7-9 Filtered, +/-30min window)\n")

    md.append("## Methodology\n")
    md.append(
        "- **Alarm universe**: unique alarm clusters from the `alarm_clusters` sheet of "
        "`DATA/1071_pvlo_alarms_clustered_with_control_actions_with_plant_context.xlsx` "
        "(`cluster_start_time`/`cluster_end_time` = alarm_start/alarm_end).\n"
        "- **FI1000 filter**: an alarm (cluster) qualifies only if **every** "
        f"`{FI1000_COL}` sample over the window `[alarm_start - 30min, alarm_end + 30min]` "
        f"falls within **[{FI1000_LOWER:.0f}, {FI1000_UPPER:.0f}]** (inclusive) — i.e. the "
        "window min >= lower AND window max <= upper. Any single excursion outside the band "
        "disqualifies the alarm.\n"
        "- **Control actions**: only `OP`/`SP` rows from the `control_actions` sheet whose "
        "`VT_Start` falls inside that SAME `[alarm_start - 30min, alarm_end + 30min]` window "
        "for a qualifying cluster are used (re-derived directly from the cluster boundaries, "
        "not assumed from the sheet's existing `before`/`during`/`after` labels).\n"
        "- **PV bucketing**: for every Source tag + action type (OP/SP), actions are grouped "
        "into up to 10 quantile-based ranges of **`03LIC_1071.PV`** (the alarm tag's PV) at the "
        "time of the action (nearest prior PV sample, backward as-of match, 5 minute tolerance). "
        "Every tag is bucketed by the SAME 1071 PV, regardless of what the operated tag itself is.\n"
        "- **Value_Range (new column)**: the min-max of the actual OP/SP level observed in "
        "that 1071-PV bucket, taken across both `PrevValue` and `Value` of the matched actions.\n"
        "- **Tag universe**: EVERY tag operated (OP/SP) in the filtered window is included — no "
        "hardcoded tag list. Tags are listed most-operated first. A tag that has no `.PV`/`.OP` "
        "column of its own is still included (bucketing uses 1071's PV, not the tag's own).\n"
    )

    print("Loading alarm clusters ...")
    clusters = load_clusters()
    print(f"Total alarm clusters: {len(clusters)}")

    print("Loading FI1000 series ...")
    fi1000_series = load_pv_series(FI1000_COL)
    clusters = compute_fi1000_stats(clusters, fi1000_series)

    no_data = int(clusters["fi1000_min"].isna().sum())
    qualifying = clusters[
        clusters["fi1000_min"].notna()
        & (clusters["fi1000_min"] >= FI1000_LOWER)
        & (clusters["fi1000_max"] <= FI1000_UPPER)
    ].copy()
    excluded_out_of_range = len(clusters) - no_data - len(qualifying)

    print(f"Clusters with no FI1000 data in window: {no_data}")
    print(f"Clusters with any FI1000 sample outside [{FI1000_LOWER},{FI1000_UPPER}]: {excluded_out_of_range}")
    print(f"Qualifying clusters (every sample in band): {len(qualifying)}")

    md.append("## Alarm (Cluster) Filtering Summary\n")
    md.append(
        f"- Total 03LIC_1071 alarm clusters: **{len(clusters)}**\n"
        f"- Excluded — no FI1000 data in window: **{no_data}**\n"
        f"- Excluded — at least one FI1000 sample outside [{FI1000_LOWER:.0f}, {FI1000_UPPER:.0f}]: "
        f"**{excluded_out_of_range}**\n"
        f"- **Qualifying alarms analyzed (every sample in band): {len(qualifying)}**\n"
    )

    window_lookup = qualifying.set_index("cluster_id")[["window_start", "window_end"]]
    qualifying_ids = set(qualifying["cluster_id"])

    print("Loading control_actions sheet ...")
    keep_cols = ["cluster_id", "cluster_start", "cluster_end", "action_timing",
                 "action_direction", "Source", "Description", "VT_Start", "PrevValue", "Value"]
    ca = pd.read_excel(WORKBOOK, sheet_name="control_actions", usecols=keep_cols)
    ca["VT_Start"] = pd.to_datetime(ca["VT_Start"])
    ca = ca[ca["cluster_id"].isin(qualifying_ids)].copy()
    ca = ca.join(window_lookup, on="cluster_id")
    ca = ca[(ca["VT_Start"] >= ca["window_start"]) & (ca["VT_Start"] <= ca["window_end"])].copy()
    print(f"Control action rows after FI1000 + +/-30min filter: {len(ca)}")

    print("Attaching 03LIC_1071.PV (the alarm tag's PV) at each action time ...")
    target_pv_series = load_pv_series(TARGET_PV_COL).rename("target_pv").reset_index()
    ca = ca.sort_values("VT_Start")
    ca = pd.merge_asof(
        ca, target_pv_series, left_on="VT_Start", right_on="TimeStamp",
        direction="backward", tolerance=ASOF_TOLERANCE,
    )

    op_sp = ca[ca["Description"].isin(["OP", "SP"])].copy()

    # ---- Action Count by Tag (ALL operated tags, most-operated first) ----
    action_counts = op_sp.groupby(["Source", "Description"]).size().unstack(fill_value=0)
    for col in ["OP", "SP"]:
        if col not in action_counts.columns:
            action_counts[col] = 0
    action_counts = action_counts[["OP", "SP"]]
    action_counts["Total"] = action_counts["OP"] + action_counts["SP"]
    action_counts = action_counts.sort_values("Total", ascending=False)

    n_no_target_pv = int(op_sp["target_pv"].isna().sum())

    md.append("## Action Count by Tag\n")
    md.append(
        "OP (Output) and SP (Setpoint) action counts for EVERY tag operated in the "
        "FI1000 7-9 filtered alarms within the +/-30min window, sorted most-operated first.\n"
    )
    ac_table = action_counts.reset_index()
    ac_table.columns = ["TagName", "OP", "SP", "Total"]
    md.append(df_to_md(ac_table))
    md.append("")

    total_op = int(action_counts["OP"].sum())
    total_sp = int(action_counts["SP"].sum())
    tol_min = int(ASOF_TOLERANCE.total_seconds() // 60)
    md.append("## Summary Statistics\n")
    md.append(
        f"- **OP actions:** {total_op}\n"
        f"- **SP actions:** {total_sp}\n"
        f"- **Total unique tags operated:** {len(action_counts)}\n"
        f"- **Actions without a 1071-PV match within {tol_min}min "
        f"(excluded from PV buckets):** {n_no_target_pv}\n"
    )

    # ---- Tag-wise analysis (every operated tag, most-operated first) ----
    md.append("## Tag-wise Analysis\n")
    md.append(
        "For each tag, step changes are bucketed by `03LIC_1071.PV` at the action time. "
        "The `PV_Range` column is therefore the 1071 level, identical in meaning for every tag.\n"
    )
    for tag in action_counts.index.tolist():
        for action_type in ["OP", "SP"]:
            if int(action_counts.loc[tag, action_type]) == 0:
                continue
            md.append(f"### TAG: {tag} — {action_type} Action Step Changes by 1071 PV Ranges\n")
            step_stats, directional, raw_count, numeric_count = analyze_tag(
                op_sp, tag, action_type
            )

            if raw_count == 0:
                md.append(f"No {action_type} actions found for this tag.\n")
                continue
            if step_stats is None or step_stats.empty:
                md.append(
                    f"{raw_count} {action_type} action(s) found, but none had both a numeric "
                    "Value/PrevValue and a 1071-PV match within tolerance.\n"
                )
                continue

            float_cols = ["Step_Min", "Step_Max", "Step_Mean", "Step_Median",
                          "Step_Std", "Step_25th", "Step_75th"]
            md.append(df_to_md(step_stats, float_cols=float_cols))
            md.append("")

            md.append(f"**{action_type} Actions — Directional Breakdown**\n")
            md.append(df_to_md(directional))
            md.append("")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSaved: {OUT_MD}")


if __name__ == "__main__":
    main()
