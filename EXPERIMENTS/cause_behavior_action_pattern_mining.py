"""
Cause-Behaviour -> Control-Action pattern mining.

Question this answers: for a group of episodes that share the SAME cause(s) AND the
SAME cause behaviour(s), did the operators apply a CONSISTENT control-action template
(same handle, same direction)? If yes, that is a rule:
    "when cause = X and it behaves like B  ->  operate handle H in direction D by ~M".

Grouping method ("explode by cause COMBINATION"):
  Only the TOP-3 ranked RCA causes are used (Cause 4/5 ignored). For each episode we take
  its top-3 causes and emit every unordered combination of size 1, 2 and 3. Combinations are
  POSITION-AGNOSTIC: a tag counts whether it was ranked 1st, 2nd or 3rd, so {A,B} and {B,A}
  land in the same group (slots are canonicalised by sorting). Each combination row carries
  that episode's full control-action matrix. We then group by (alarm_tag, combination) and,
  for every control handle operated on that alarm, measure:
      operated_frac   = in what fraction of the group's episodes was the handle moved
      dir_consistency = of the directional moves, what fraction went the SAME way
      avg_step        = mean move size
  A handle is a TEMPLATE action for the group when it is moved often AND consistently.

Running combo sizes 1 / 2 / 3 side by side shows whether fixing a 2nd (and 3rd) co-occurring
cause makes the operator response more repeatable than knowing a single cause alone.

READ-ONLY on the source workbook. Writes a separate results workbook.
"""
from pathlib import Path
from itertools import combinations
from collections import Counter
import re
import numpy as np
import pandas as pd

SRC_XLSX = Path("/home/h604827/ControlActions/DATA/RCA_cause_behavior_control_actions_ALL_TAGS.xlsx")
OUT_XLSX = Path("/home/h604827/ControlActions/DATA/RCA_cause_behavior_action_patterns.xlsx")

N_CAUSES = 3                                              # use Cause 1..3 only
CAUSE_COLS = [f"Cause {i}" for i in range(1, N_CAUSES + 1)]
GROUP_SIZES = (1, 2, 3)

# ── Template thresholds (tune here) ────────────────────────────────────────────
MIN_EPISODES = 4      # a group needs at least this many episodes
FREQ_MIN     = 0.50   # handle moved in >= this fraction of the group's episodes
CONS_MIN     = 0.75   # of the inc/dec moves, >= this fraction go the same way
MIN_DIR_N    = 4      # need at least this many inc/dec moves to trust the direction

# behaviour detail levels: which column supplies the 2nd key of each cause slot
GRANULARITIES = {"tag only": None, "tag + pattern": "pattern", "tag + behaviour": "behavior"}

SLOT_COLS = [c for i in GROUP_SIZES for c in (f"cause_{i}", f"behavior_{i}")]

# exported column sets (trimmed to what the SMEs actually read); behaviour is split per slot
SLOT_OUT_COLS = [c for i in GROUP_SIZES
                 for c in (f"cause_{i}", f"pattern_{i}", f"direction_{i}", f"roc_{i}")]
_GROUP_ID = ["granularity", "group_size", "alarm_tag"] + SLOT_OUT_COLS + ["group_label", "n_episodes", "episodes"]
HANDLE_COLS = _GROUP_ID + ["handle", "n_operated", "operated_frac", "n_inc", "n_dec",
                           "n_nochg", "dominant_dir", "dir_consistency",
                           "avg_step", "min_step", "max_step", "template"]
TEMPLATE_COLS = HANDLE_COLS + ["strength"]
REC_COLS = _GROUP_ID + ["n_template_handles", "has_template", "best_handle", "best_action",
                        "best_operated_frac", "best_avg_step", "best_min_step", "best_max_step"]

_CAUSE_RE = re.compile(r'^\s*(\S+\.PV)\s*\[\s*(.*?)\s*\|\s*(.*?)\s*\|\s*(.*?)\s*\]\s*$')
_ARROW = {"increase": "up", "decrease": "down"}


def parse_cause(v):
    if not isinstance(v, str):
        return None
    m = _CAUSE_RE.match(v)
    return m.groups() if m else None      # (tag, pattern, direction, roc)


def _label(cause, beh):
    return f"{cause} [{beh}]" if beh else cause


def split_slots(df):
    """Expand each slot's behaviour string into pattern / direction / roc columns."""
    df = df.copy()
    for i in GROUP_SIZES:
        parts = df[f"behavior_{i}"].fillna("").str.split(" | ", regex=False, expand=True)
        for j, name in enumerate(("pattern", "direction", "roc")):
            df[f"{name}_{i}"] = parts[j].fillna("") if j in parts.columns else ""
    return df


def load_long():
    """Explode every episode into all size-1/2/3 combinations of its top-3 causes.

    The cause universe is the union of the unique tags seen in Cause 1/2/3. A group of size k
    is one unordered set of k of those tags; an episode belongs to the group when all k tags
    appear anywhere in its top-3 causes. So size 1 is the superset that sizes 2 and 3 split.
    """
    xl = pd.ExcelFile(SRC_XLSX)
    sheets, rows, universe = {}, [], []
    for sh in xl.sheet_names:
        d = pd.read_excel(SRC_XLSX, sheet_name=sh)
        sheets[sh] = dict(df=d, dir_cols=[c for c in d.columns if c.endswith(" dir")])

        ep_causes = {}
        for idx, row in d.iterrows():
            causes, seen = [], set()
            for cc in CAUSE_COLS:
                p = parse_cause(row[cc])
                if p is None or p[0] in seen:      # keep the highest-ranked hit per tag
                    continue
                seen.add(p[0])
                tag, pattern, direction, roc = p
                causes.append(dict(cause=tag, pattern=pattern,
                                   behavior=f"{pattern} | {direction} | {roc}"))
            ep_causes[idx] = (row["Episode number"], causes)

        # slot order = rarest tag first, so the DISCRIMINATING cause leads and a
        # near-ubiquitous one (e.g. 02FI_1000) falls to the last slot instead of always cause_1
        freq = Counter(c["cause"] for _, cs in ep_causes.values() for c in cs)
        universe += [dict(alarm_tag=sh, cause_tag=t, n_episodes_with_tag=n,
                          pct_episodes=round(100 * n / len(d), 1)) for t, n in freq.items()]

        for idx, (ep_no, causes) in ep_causes.items():
            for k in GROUP_SIZES:
                for combo in combinations(causes, k):
                    # canonical order -> {A,B} and {B,A} land in the same group
                    combo = sorted(combo, key=lambda c: (freq[c["cause"]], c["cause"]))
                    rec = dict(alarm_tag=sh, ep_idx=idx, episode=ep_no, group_size=k)
                    for i in GROUP_SIZES:
                        c = combo[i - 1] if i <= k else dict(cause="", pattern="", behavior="")
                        rec[f"cause_{i}"] = c["cause"]
                        rec[f"pattern_{i}"] = c["pattern"]
                        rec[f"behavior_{i}"] = c["behavior"]
                    rows.append(rec)
    return pd.DataFrame(rows), sheets, pd.DataFrame(universe)


def key_cols(k, beh_level):
    """Group-key columns for a group of size k at a given behaviour detail level."""
    cols = []
    for i in range(1, k + 1):
        cols.append(f"cause_{i}")
        if beh_level:
            cols.append(f"{beh_level}_{i}")
    return cols


def analyse(long, sheets, k, beh_level, granularity):
    """Score every control handle's consistency inside each (alarm_tag, cause-group) group."""
    gk = key_cols(k, beh_level)
    recs = []
    for alarm_tag, sub in long[long["group_size"] == k].groupby("alarm_tag"):
        d, dir_cols = sheets[alarm_tag]["df"], sheets[alarm_tag]["dir_cols"]
        for gvals, g in sub.groupby(gk, sort=False):
            ep_idxs = g["ep_idx"].unique()          # one episode counted once per group
            n = len(ep_idxs)
            if n < MIN_EPISODES:
                continue
            sub_d = d.loc[ep_idxs]
            # episode numbers as printed in the source workbook, for cross-checking
            ep_list = ", ".join(str(e) for e in sorted(g.drop_duplicates("ep_idx")["episode"]))
            gvals = gvals if isinstance(gvals, tuple) else (gvals,)
            gdict = dict(zip(gk, gvals))
            slots = {}
            for i in GROUP_SIZES:
                slots[f"cause_{i}"] = gdict.get(f"cause_{i}", "")
                slots[f"behavior_{i}"] = gdict.get(f"{beh_level}_{i}", "") if beh_level else ""
            group_label = " + ".join(_label(slots[f"cause_{i}"], slots[f"behavior_{i}"])
                                     for i in range(1, k + 1))
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
                mag = pd.to_numeric(sub_d[mag_col], errors="coerce").dropna()   # per-episode avg step
                is_template = bool(operated_frac >= FREQ_MIN and dir_n >= MIN_DIR_N
                                   and dir_cons >= CONS_MIN)
                recs.append(dict(granularity=granularity, group_size=k, alarm_tag=alarm_tag,
                                 **slots, group_label=group_label, n_episodes=n, episodes=ep_list,
                                 handle=h[:-4], n_operated=n_op,
                                 operated_frac=round(operated_frac, 2),
                                 n_inc=n_inc, n_dec=n_dec, n_nochg=n_nc,
                                 dominant_dir=dominant,
                                 dir_consistency=round(dir_cons, 2) if dir_n else np.nan,
                                 avg_step=round(mag.mean(), 2) if len(mag) else np.nan,
                                 min_step=round(mag.min(), 2) if len(mag) else np.nan,
                                 max_step=round(mag.max(), 2) if len(mag) else np.nan,
                                 template=is_template,
                                 strength=round(operated_frac * (dir_cons if dir_n else 0), 3)))
    return pd.DataFrame(recs)


def per_group(res):
    """Collapse handle rows to one row per group, keeping the strongest template handle."""
    if res.empty:
        return pd.DataFrame()
    idx = ["granularity", "group_size", "alarm_tag"] + SLOT_COLS
    rows = []
    for keys, g in res.groupby(idx, sort=False):
        tmpl = g[g["template"]].sort_values("strength", ascending=False)
        best = tmpl.iloc[0] if len(tmpl) else None
        rows.append(dict(zip(idx, keys),
                         group_label=g["group_label"].iloc[0],
                         n_episodes=int(g["n_episodes"].iloc[0]),
                         episodes=g["episodes"].iloc[0],
                         n_template_handles=int(g["template"].sum()),
                         has_template=bool(g["template"].any()),
                         best_handle=None if best is None else best["handle"],
                         best_action=None if best is None else _ARROW[best["dominant_dir"]],
                         best_operated_frac=None if best is None else best["operated_frac"],
                         best_consistency=None if best is None else best["dir_consistency"],
                         best_avg_step=None if best is None else best["avg_step"],
                         best_min_step=None if best is None else best["min_step"],
                         best_max_step=None if best is None else best["max_step"],
                         best_strength=0.0 if best is None else best["strength"],
                         best_any_strength=round(g["strength"].max(), 3)))
    return pd.DataFrame(rows)


def coverage(universe, groups_all):
    """Per cause tag: how many qualifying groups it reaches, and which slot it lands in."""
    gg = groups_all[groups_all["granularity"].str.endswith("tag only")]
    rows = []
    for _, u in universe.iterrows():
        r = dict(u)
        for k in GROUP_SIZES:
            sub = gg[(gg["group_size"] == k) & (gg["alarm_tag"] == u["alarm_tag"])]
            in_any = sub[[f"cause_{i}" for i in GROUP_SIZES]].eq(u["cause_tag"]).any(axis=1)
            r[f"groups_size{k}"] = int(in_any.sum())
        for i in GROUP_SIZES:
            r[f"in_slot_{i}"] = int((gg[gg["alarm_tag"] == u["alarm_tag"]][f"cause_{i}"]
                                     == u["cause_tag"]).sum())
        rows.append(r)
    return (pd.DataFrame(rows)
            .sort_values(["alarm_tag", "n_episodes_with_tag"], ascending=[True, False])
            .reset_index(drop=True))


def main():
    long, sheets, universe = load_long()
    singles = long[long["group_size"] == 1]
    n_eps = int(singles.groupby("alarm_tag")["ep_idx"].nunique().sum())  # ep_idx repeats per sheet
    print(f"Loaded {long['alarm_tag'].nunique()} alarm sheets | {n_eps} episodes | "
          f"top-{N_CAUSES} causes -> {len(long)} cause-group observations")
    print(f"Cause universe: {len(universe)} (alarm, cause tag) pairs | "
          f"{universe['cause_tag'].nunique()} distinct cause tags\n")

    # how many groups clear MIN_EPISODES at each group size / behaviour detail
    print("=" * 92)
    print(f"  GROUPS AVAILABLE  (qualifying / total, qualifying = >= {MIN_EPISODES} episodes)")
    print("=" * 92)
    avail = []
    for k in GROUP_SIZES:
        sub = long[long["group_size"] == k]
        row = {"group_size": k}
        for gname, beh in GRANULARITIES.items():
            sizes = sub.groupby(["alarm_tag"] + key_cols(k, beh))["ep_idx"].nunique()
            row[gname] = f"{int((sizes >= MIN_EPISODES).sum())} / {len(sizes)}"
        avail.append(row)
    print(pd.DataFrame(avail).to_string(index=False))

    # ── run every group size x behaviour detail level ───────────────────────────
    all_res, comp = [], []
    for k in GROUP_SIZES:
        for gname, beh in GRANULARITIES.items():
            label = f"{k} cause{'s' if k > 1 else ''} | {gname}"
            res = analyse(long, sheets, k, beh, label)
            all_res.append(res)
            grp = per_group(res)
            comp.append(dict(granularity=label, group_size=k, behaviour_detail=gname,
                             n_groups=len(grp),
                             n_with_template=int(grp["has_template"].sum()) if len(grp) else 0,
                             pct_with_template=round(100 * grp["has_template"].mean(), 1) if len(grp) else 0.0,
                             median_best_strength=round(grp["best_any_strength"].median(), 3) if len(grp) else np.nan))
    res_all = pd.concat([r for r in all_res if len(r)], ignore_index=True)
    comp = pd.DataFrame(comp)
    groups_all = per_group(res_all)
    cov = coverage(universe, groups_all)

    print("\n" + "=" * 92)
    print("  DOES A CONSISTENT ACTION TEMPLATE EXIST PER GROUP?  (higher pct = more rule-like)")
    print("=" * 92)
    print(comp.to_string(index=False))

    templates = (res_all[res_all["template"]]
                 .sort_values(["group_size", "strength"], ascending=[True, False])
                 .reset_index(drop=True))

    # ── SME-facing: strongest templates for each group size ─────────────────────
    for k in GROUP_SIZES:
        t = templates[(templates["group_size"] == k) &
                      (templates["granularity"].str.endswith("behaviour"))]
        print("\n" + "=" * 92)
        print(f"  TOP TEMPLATES | group size {k} + full behaviour   (found {len(t)})")
        print("=" * 92)
        if t.empty:
            print("  none -- not enough episodes share this cause group")
            continue
        for _, r in t.sort_values("strength", ascending=False).head(10).iterrows():
            print(f"[{r['alarm_tag']}] {r['group_label']}  (n={r['n_episodes']})\n"
                  f"      -> {r['handle']}: move {_ARROW[r['dominant_dir']].upper():4s} "
                  f"in {r['operated_frac']:.0%} of eps, {r['dir_consistency']:.0%} consistent, "
                  f"avg step {r['avg_step']}  (strength {r['strength']})")

    print("\n" + "=" * 92)
    print("  VERDICT (full-behaviour grouping)")
    print("=" * 92)
    for _, r in comp[comp["behaviour_detail"] == "tag + behaviour"].iterrows():
        print(f"  group size {r['group_size']}: {r['n_with_template']}/{r['n_groups']} groups have "
              f">=1 consistent template action  ({r['pct_with_template']:.0f}%)")

    print("\n  slot occupancy (tag-only granularity, all group sizes):")
    for i in GROUP_SIZES:
        print(f"    cause_{i}: {int((cov[f'in_slot_{i}'] > 0).sum())} distinct tags ever placed here")

    # ── write results (separate workbook; source untouched) ──────────────────────
    handles = res_all.sort_values(["group_size", "granularity", "alarm_tag",
                                   "cause_1", "cause_2", "cause_3", "strength"],
                                  ascending=[True, True, True, True, True, True, False])
    recs = groups_all.sort_values(["group_size", "has_template", "best_strength", "n_episodes"],
                                  ascending=[True, False, False, False])
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        split_slots(handles)[HANDLE_COLS].to_excel(w, sheet_name="all_group_handles", index=False)
        split_slots(recs)[REC_COLS].to_excel(w, sheet_name="group_recommendations", index=False)
        split_slots(templates)[TEMPLATE_COLS].to_excel(w, sheet_name="templates", index=False)
        cov.to_excel(w, sheet_name="cause_universe", index=False)
    print(f"\nWrote -> {OUT_XLSX}")
    print("  sheets: all_group_handles (main) | group_recommendations | templates | cause_universe")


if __name__ == "__main__":
    main()

    print(f"\nWrote -> {OUT_XLSX}")
    print("  sheets: granularity_compare | group_recommendations | templates_full_behavior | all_group_handles")


if __name__ == "__main__":
    main()
