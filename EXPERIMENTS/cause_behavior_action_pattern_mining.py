"""
Cause-Behaviour -> Control-Action pattern mining.

Question this answers: for a group of episodes that share the SAME cause tag AND the
SAME cause behaviour, did the operators apply a CONSISTENT control-action template
(same handle, same direction)? If yes, that is a rule:
    "when cause = X and it behaves like B  ->  operate handle H in direction D by ~M".

Grouping method ("explode by cause"):
  Each episode has 5 ranked RCA causes. We melt them so one episode contributes up to
  5 (cause_tag, behaviour) observations, EACH carrying that episode's full control-action
  matrix. We then group by (alarm_tag, cause_tag, behaviour) and, for every control handle
  operated on that alarm, measure:
      operated_frac   = in what fraction of the group's episodes was the handle moved
      dir_consistency = of the directional moves, what fraction went the SAME way
      avg_step        = mean move size
  A handle is a TEMPLATE action for the group when it is moved often AND consistently.

READ-ONLY on the source workbook. Writes a separate results workbook.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd

SRC_XLSX = Path("/home/h604827/ControlActions/DATA/RCA_cause_behavior_control_actions_ALL_TAGS.xlsx")
OUT_XLSX = Path("/home/h604827/ControlActions/DATA/RCA_cause_behavior_action_patterns.xlsx")

CAUSE_COLS = [f"Cause {i}" for i in range(1, 6)]

# ── Template thresholds (tune here) ────────────────────────────────────────────
MIN_EPISODES = 8      # a (cause,behaviour) group needs at least this many episodes
FREQ_MIN     = 0.50   # handle moved in >= this fraction of the group's episodes
CONS_MIN     = 0.75   # of the inc/dec moves, >= this fraction go the same way
MIN_DIR_N    = 4      # need at least this many inc/dec moves to trust the direction

_CAUSE_RE = re.compile(r'^\s*(\S+\.PV)\s*\[\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\]\s*$')
_ARROW = {"increase": "up", "decrease": "down"}


def parse_cause(v):
    if not isinstance(v, str):
        return None
    m = _CAUSE_RE.match(v)
    return m.groups() if m else None      # (tag, pattern, direction, roc)


def load_long():
    """Explode all sheets into long form + keep each sheet frame and its handle columns."""
    xl = pd.ExcelFile(SRC_XLSX)
    sheets, rows = {}, []
    for sh in xl.sheet_names:
        d = pd.read_excel(SRC_XLSX, sheet_name=sh)
        sheets[sh] = dict(df=d, dir_cols=[c for c in d.columns if c.endswith(" dir")])
        for idx, row in d.iterrows():
            for slot, cc in enumerate(CAUSE_COLS, 1):
                p = parse_cause(row[cc])
                if p is None:
                    continue
                tag, pattern, direction, roc = p
                rows.append(dict(alarm_tag=sh, ep_idx=idx, episode=row["Episode number"],
                                 cause_slot=slot, cause_tag=tag, pattern=pattern,
                                 direction=direction, roc=roc,
                                 behavior=f"{pattern} | {direction} | {roc}"))
    return pd.DataFrame(rows), sheets


def analyse(long, sheets, group_keys):
    """Score every control handle's consistency inside each (alarm_tag, *group_keys) group."""
    recs = []
    for alarm_tag, sub in long.groupby("alarm_tag"):
        d, dir_cols = sheets[alarm_tag]["df"], sheets[alarm_tag]["dir_cols"]
        for gvals, g in sub.groupby(group_keys):
            ep_idxs = g["ep_idx"].unique()          # one episode counted once per group
            n = len(ep_idxs)
            if n < MIN_EPISODES:
                continue
            sub_d = d.loc[ep_idxs]
            gvals = gvals if isinstance(gvals, tuple) else (gvals,)
            gdict = dict(zip(group_keys, gvals))
            for h in dir_cols:
                col = sub_d[h]
                n_inc = int((col == "increased").sum())
                n_dec = int((col == "decreased").sum())
                n_nc = int((col == "no change").sum())
                n_op = n_inc + n_dec + n_nc
                if n_op == 0:
                    continue
                dir_n = n_inc + n_dec
                operated_frac = n_op / n
                dir_cons = (max(n_inc, n_dec) / dir_n) if dir_n else np.nan
                dominant = "increase" if n_inc >= n_dec else "decrease"
                mag_col = h[:-4] + " avg|step|"
                mag_mean = pd.to_numeric(sub_d.get(mag_col), errors="coerce").mean()
                is_template = bool(operated_frac >= FREQ_MIN and dir_n >= MIN_DIR_N
                                   and dir_cons >= CONS_MIN)
                recs.append(dict(alarm_tag=alarm_tag, **gdict, n_episodes=n,
                                 handle=h[:-4], n_operated=n_op,
                                 operated_frac=round(operated_frac, 2),
                                 n_inc=n_inc, n_dec=n_dec, n_nochg=n_nc,
                                 dominant_dir=dominant,
                                 dir_consistency=round(dir_cons, 2) if dir_n else np.nan,
                                 avg_step=round(mag_mean, 2) if pd.notna(mag_mean) else np.nan,
                                 template=is_template,
                                 strength=round(operated_frac * (dir_cons if dir_n else 0), 3)))
    return pd.DataFrame(recs)


def summarize(res, group_keys):
    """Per-granularity headline: how many groups have >=1 template handle."""
    if res.empty:
        return dict(n_groups=0, n_with_template=0, pct_with_template=0.0, median_best_strength=np.nan)
    gcols = ["alarm_tag"] + list(group_keys)
    per = res.groupby(gcols).agg(has_t=("template", "any"), best=("strength", "max"))
    return dict(n_groups=len(per), n_with_template=int(per["has_t"].sum()),
                pct_with_template=round(100 * per["has_t"].mean(), 1),
                median_best_strength=round(per["best"].median(), 3))


def main():
    long, sheets = load_long()
    n_eps = int(long.groupby("alarm_tag")["ep_idx"].nunique().sum())   # ep_idx repeats across sheets
    print(f"Loaded {long['alarm_tag'].nunique()} alarm sheets | "
          f"{n_eps} episodes exploded into {len(long)} cause-observations\n")

    # group-size sanity: how many (cause, full-behaviour) groups clear MIN_EPISODES?
    gk_full = ["cause_tag", "pattern", "direction", "roc"]
    sizes = (long.groupby(["alarm_tag"] + gk_full)["ep_idx"].nunique())
    print(f"(cause,behaviour) groups total: {len(sizes)} | "
          f"with >= {MIN_EPISODES} episodes: {(sizes >= MIN_EPISODES).sum()}\n")

    # ── run the three granularities to see whether behaviour adds signal ──────────
    gk_cause = ["cause_tag"]
    gk_pat = ["cause_tag", "pattern"]
    res_cause = analyse(long, sheets, gk_cause)
    res_pat = analyse(long, sheets, gk_pat)
    res_full = analyse(long, sheets, gk_full)

    print("=" * 78)
    print("  DOES A CONSISTENT ACTION TEMPLATE EXIST PER GROUP?  (higher pct = more rule-like)")
    print("=" * 78)
    comp = pd.DataFrame([
        dict(granularity="cause only",            **summarize(res_cause, gk_cause)),
        dict(granularity="cause + pattern",       **summarize(res_pat, gk_pat)),
        dict(granularity="cause + full behaviour", **summarize(res_full, gk_full)),
    ])
    print(comp.to_string(index=False))

    # ── the SME-facing template list (full behaviour, only the consistent handles) ─
    templates = (res_full[res_full["template"]]
                 .sort_values(["alarm_tag", "strength"], ascending=[True, False])
                 .reset_index(drop=True))
    print("\n" + "=" * 78)
    print(f"  TEMPLATES FOUND (full behaviour, template=True): {len(templates)}")
    print("=" * 78)
    top = templates.sort_values("strength", ascending=False).head(20)
    for _, r in top.iterrows():
        print(f"[{r['alarm_tag']}] cause {r['cause_tag']} [{r['pattern']} | {r['direction']} | {r['roc']}]"
              f" (n={r['n_episodes']})\n"
              f"      -> {r['handle']}: move {_ARROW[r['dominant_dir']].upper():4s} "
              f"in {r['operated_frac']:.0%} of eps, {r['dir_consistency']:.0%} consistent, "
              f"avg step {r['avg_step']}  (strength {r['strength']})")

    # ── one-row-per-group recommendation summary ─────────────────────────────────
    grp_rows = []
    for keys, g in res_full.groupby(["alarm_tag"] + gk_full):
        alarm_tag, cause_tag, pattern, direction, roc = keys
        tmpl = g[g["template"]].sort_values("strength", ascending=False)
        best = tmpl.iloc[0] if len(tmpl) else None
        grp_rows.append(dict(
            alarm_tag=alarm_tag, cause_tag=cause_tag,
            behavior=f"{pattern} | {direction} | {roc}",
            n_episodes=int(g["n_episodes"].iloc[0]),
            n_template_handles=int(g["template"].sum()),
            has_template=bool(g["template"].any()),
            best_handle=None if best is None else best["handle"],
            best_action=None if best is None else _ARROW[best["dominant_dir"]],
            best_operated_frac=None if best is None else best["operated_frac"],
            best_consistency=None if best is None else best["dir_consistency"],
            best_avg_step=None if best is None else best["avg_step"],
        ))
    group_summary = (pd.DataFrame(grp_rows)
                     .sort_values(["has_template", "n_template_handles", "n_episodes"],
                                  ascending=[False, False, False])
                     .reset_index(drop=True))

    n_grp = len(group_summary)
    n_hit = int(group_summary["has_template"].sum())
    print("\n" + "=" * 78)
    print(f"  VERDICT: {n_hit}/{n_grp} qualifying (cause,behaviour) groups have >=1 consistent "
          f"template action  ({100*n_hit/max(n_grp,1):.0f}%)")
    print("=" * 78)

    # ── write results (separate workbook; source untouched) ──────────────────────
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        comp.to_excel(w, sheet_name="granularity_compare", index=False)
        group_summary.to_excel(w, sheet_name="group_recommendations", index=False)
        templates.to_excel(w, sheet_name="templates_full_behavior", index=False)
        res_full.sort_values(["alarm_tag", "cause_tag", "strength"], ascending=[True, True, False]) \
                 .to_excel(w, sheet_name="all_group_handles", index=False)
    print(f"\nWrote -> {OUT_XLSX}")
    print("  sheets: granularity_compare | group_recommendations | templates_full_behavior | all_group_handles")


if __name__ == "__main__":
    main()
