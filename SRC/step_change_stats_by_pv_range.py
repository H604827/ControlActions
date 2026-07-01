"""
Step-change statistics of control actions, bucketed by 03LIC_1071 PV range.

For every control action taken DURING an alarm (action_timing == 'during'),
look up the 03LIC_1071.PV value the operator was reacting to (PV at/just before
the action timestamp), then bucket the actions into PV ranges and summarise the
step-change magnitude (Value - PrevValue).

Two tables are produced:
  * OP  -> Description == 'OP'
  * SP  -> Description == 'SP'

Output: an Excel workbook with one sheet per action type, plus console print.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[1]
WORKBOOK = (
    REPO
    / "RESULTS/03LIC_1071_PVLO_episodes_12JUN2026_1219"
    / "03LIC_1071_pvlo_alarms_clustered_with_control_actions.xlsx"
)
PV_PARQUET = REPO / "DATA/PV-OP_data/03LIC_1071_JAN_2026.parquet"
PV_COL = "03LIC_1071.PV"
OUT_XLSX = WORKBOOK.parent / "step_change_stats_by_pv_range_during_alarm.xlsx"

N_BINS = 10                     # quantile bins (like the reference table)
ASOF_TOLERANCE = pd.Timedelta("5min")   # don't match across long gaps / trips


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def attach_target_pv(actions: pd.DataFrame, pv: pd.DataFrame) -> pd.DataFrame:
    """Attach the 03LIC_1071 PV the operator saw when each action was taken."""
    actions = actions.sort_values("VT_Start").reset_index(drop=True)
    pv = pv[["TimeStamp", PV_COL]].dropna().sort_values("TimeStamp")
    merged = pd.merge_asof(
        actions,
        pv,
        left_on="VT_Start",
        right_on="TimeStamp",
        direction="backward",       # state the operator observed before acting
        tolerance=ASOF_TOLERANCE,
    )
    return merged.rename(columns={PV_COL: "target_pv"})


def step_stats_by_pv_range(df: pd.DataFrame, n_bins: int = N_BINS) -> pd.DataFrame:
    """Bucket actions by target PV (quantile bins) and summarise step change."""
    work = df.dropna(subset=["target_pv", "step_change"]).copy()
    work["PV_Range"] = pd.qcut(work["target_pv"], q=n_bins, duplicates="drop")

    grouped = work.groupby("PV_Range", observed=True)["step_change"]
    stats = grouped.agg(
        Data_Points="count",
        Step_Min="min",
        Step_Max="max",
        Step_Mean="mean",
        Step_Median="median",
        Step_Std="std",
        Step_25th=lambda s: s.quantile(0.25),
        Step_75th=lambda s: s.quantile(0.75),
    ).reset_index()

    # Round PV interval edges for readability, mirroring the reference table.
    stats["PV_Range"] = stats["PV_Range"].apply(
        lambda iv: f"({iv.left:.3g}, {iv.right:.3g}]"
    )
    return stats


def build_table(actions_during: pd.DataFrame, pv: pd.DataFrame, desc: str) -> pd.DataFrame:
    sub = actions_during[actions_during["Description"] == desc].copy()
    # Keep only numeric step changes (drops discrete ON/OFF style "OP" rows).
    sub["PrevValue"] = pd.to_numeric(sub["PrevValue"], errors="coerce")
    sub["Value"] = pd.to_numeric(sub["Value"], errors="coerce")
    sub = sub.dropna(subset=["PrevValue", "Value"])
    sub["step_change"] = sub["Value"] - sub["PrevValue"]

    sub = attach_target_pv(sub, pv)
    n_no_pv = sub["target_pv"].isna().sum()
    table = step_stats_by_pv_range(sub)

    print(f"\n[{desc}] during-alarm numeric actions: {len(sub):,} "
          f"(dropped {n_no_pv:,} without a PV match within {ASOF_TOLERANCE})")
    return table


def fmt(table: pd.DataFrame) -> str:
    show = table.copy()
    for c in ["Step_Mean", "Step_Median", "Step_Std", "Step_25th", "Step_75th",
              "Step_Min", "Step_Max"]:
        show[c] = show[c].map(lambda x: f"{x:.4f}")
    return show.to_string(index=False)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    ca = pd.read_excel(WORKBOOK, sheet_name="control_actions")
    ca["VT_Start"] = pd.to_datetime(ca["VT_Start"])
    during = ca[ca["action_timing"] == "during"].copy()
    print(f"Total control actions: {len(ca):,} | during-alarm: {len(during):,}")

    pv = pd.read_parquet(PV_PARQUET, columns=["TimeStamp", PV_COL])
    pv["TimeStamp"] = pd.to_datetime(pv["TimeStamp"])

    op_table = build_table(during, pv, "OP")
    sp_table = build_table(during, pv, "SP")

    print("\n" + "=" * 78)
    print("03LIC_1071 — OP Action Step Changes by PV Ranges (during alarm)")
    print("=" * 78)
    print(fmt(op_table))

    print("\n" + "=" * 78)
    print("03LIC_1071 — SP Action Step Changes by PV Ranges (during alarm)")
    print("=" * 78)
    print(fmt(sp_table))

    with pd.ExcelWriter(OUT_XLSX) as writer:
        op_table.to_excel(writer, sheet_name="OP_step_changes", index=False)
        sp_table.to_excel(writer, sheet_name="SP_step_changes", index=False)
    print(f"\nSaved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
