"""
Filter 03LIC_1071 PVLO alarm episodes by a 3-tag "something bad happened" condition
and export the windowed control actions + stats for the passing episodes.

Condition (all three must be reached somewhere in the window — "Higher" limits):
    02FI_1000.PV  >= 8.5
    03TI_1421.PV  >= -3
    03TI_1901.PV  >= -31

Window per episode (cluster):
    [cluster_start_time - 240 min (4 h before), cluster_end_time + 60 min (1 h after)]

Outputs an Excel with:
    - passing_episodes          : one row per passing cluster + evidence (max tag values)
    - control_actions           : base control actions clipped to the window, passing clusters
    - step_stats_by_tag_action  : recomputed step statistics per (Tag, ActionType)
"""
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──
RUN = Path('/home/h604827/ControlActions/RESULTS/03LIC_1071_PVLO_episodes_12JUN2026_1219')
CLUSTERS_FILE = RUN / '03LIC_1071_pvlo_alarms_clustered_with_control_actions.xlsx'
PV_OP_FILE = '/home/h604827/ControlActions/DATA/PV-OP_data/03LIC_1071_JAN_2026.parquet'
TRIP_FILE = '/home/h604827/ControlActions/DATA/Final_List_Trip_Duration.csv'
OUT_FILE = RUN / '03LIC_1071_pvlo_3tag_episodes_control_actions_4hr_1hr_window.xlsx'

# ── Condition config ──
CONDITIONS = {
    '02FI_1000.PV': 8.5,    # FI1000 Higher limit
    '03TI_1421.PV': -3.0,   # TI1421 Higher limit
    '03TI_1901.PV': -31.0,  # TI1901 Higher limit
}
BUFFER_BEFORE_MIN = 240  # 4 h before alarm
BUFFER_AFTER_MIN = 60    # 1 h after alarm
FILTER_TRIPS = True

# ════════════════════════════════════════════════════════════════════════════
# 1. Load alarm clusters (539 episodes) -> one row per cluster
# ════════════════════════════════════════════════════════════════════════════
alarms_df = pd.read_excel(CLUSTERS_FILE, sheet_name='alarm_clusters')
alarms_df['cluster_start_time'] = pd.to_datetime(alarms_df['cluster_start_time'])
alarms_df['cluster_end_time'] = pd.to_datetime(alarms_df['cluster_end_time'])

clusters = (alarms_df
            .groupby('cluster_id')
            .agg(cluster_start_time=('cluster_start_time', 'first'),
                 cluster_end_time=('cluster_end_time', 'first'),
                 cluster_type=('cluster_type', 'first'),
                 cluster_total_alarms=('cluster_total_alarms', 'first'),
                 cluster_total_duration_min=('cluster_total_duration_min', 'first'))
            .reset_index())
print(f'Total episodes (clusters): {len(clusters)}')

# ════════════════════════════════════════════════════════════════════════════
# 2. Load PV/OP data, trip-filter, keep only the 3 condition tags
# ════════════════════════════════════════════════════════════════════════════
pv = pd.read_parquet(PV_OP_FILE, columns=['TimeStamp'] + list(CONDITIONS))
pv['TimeStamp'] = pd.to_datetime(pv['TimeStamp'])
pv = pv.set_index('TimeStamp').sort_index()

if FILTER_TRIPS:
    trips = pd.read_csv(TRIP_FILE)
    trips['Stop Date'] = pd.to_datetime(trips['Stop Date'])
    trips['Start Date'] = pd.to_datetime(trips['Start Date'])
    pre = len(pv)
    trip_mask = pd.Series(False, index=pv.index)
    for _, t in trips.iterrows():
        trip_mask |= (pv.index >= t['Stop Date']) & (pv.index <= t['Start Date'])
    pv = pv[~trip_mask]
    print(f'Trip filtering (PV): {pre:,} -> {len(pv):,} rows (removed {pre - len(pv):,})')

for tag in CONDITIONS:
    pv[tag] = pd.to_numeric(pv[tag], errors='coerce')

# ════════════════════════════════════════════════════════════════════════════
# 3. Evaluate the 3-tag condition per episode over its 4 h / 1 h window
# ════════════════════════════════════════════════════════════════════════════
rows = []
for r in clusters.itertuples(index=False):
    w_start = r.cluster_start_time - pd.Timedelta(minutes=BUFFER_BEFORE_MIN)
    w_end = r.cluster_end_time + pd.Timedelta(minutes=BUFFER_AFTER_MIN)
    win = pv.loc[w_start:w_end]

    rec = {
        'cluster_id': r.cluster_id,
        'cluster_start_time': r.cluster_start_time,
        'cluster_end_time': r.cluster_end_time,
        'cluster_type': r.cluster_type,
        'cluster_total_alarms': r.cluster_total_alarms,
        'cluster_total_duration_min': r.cluster_total_duration_min,
        'window_start': w_start,
        'window_end': w_end,
        'n_pv_samples': len(win),
    }
    passes_all = True
    for tag, thr in CONDITIONS.items():
        mx = win[tag].max()  # NaN if no/empty data -> condition fails
        short = tag.split('.')[0]
        rec[f'{short}_max'] = mx
        cond = bool(pd.notna(mx) and mx >= thr)
        rec[f'{short}_pass'] = cond
        passes_all = passes_all and cond
    rec['passes_all_3'] = passes_all
    rows.append(rec)

episodes = pd.DataFrame(rows)
passing = episodes[episodes['passes_all_3']].copy()
passing_ids = set(passing['cluster_id'])

print('\n' + '=' * 70)
print('PER-CONDITION PASS COUNTS (independently, over the 4h/1h window)')
print('=' * 70)
for tag in CONDITIONS:
    short = tag.split('.')[0]
    print(f'  {short} >= {CONDITIONS[tag]:>6}: {episodes[f"{short}_pass"].sum():>4} / {len(episodes)} episodes')
print(f'\n  ALL THREE conditions met : {len(passing)} / {len(episodes)} episodes')

# ════════════════════════════════════════════════════════════════════════════
# 4. Control actions: base sheet -> passing clusters -> clip to 4h/1h window
# ════════════════════════════════════════════════════════════════════════════
actions = pd.read_excel(CLUSTERS_FILE, sheet_name='control_actions')
actions['VT_Start'] = pd.to_datetime(actions['VT_Start'])
actions['cluster_start'] = pd.to_datetime(actions['cluster_start'])
actions['cluster_end'] = pd.to_datetime(actions['cluster_end'])

act = actions[actions['cluster_id'].isin(passing_ids)].copy()
win_lo = act['cluster_start'] - pd.Timedelta(minutes=BUFFER_BEFORE_MIN)
win_hi = act['cluster_end'] + pd.Timedelta(minutes=BUFFER_AFTER_MIN)
act = act[(act['VT_Start'] >= win_lo) & (act['VT_Start'] <= win_hi)].copy()
act = act.sort_values(['cluster_id', 'VT_Start']).reset_index(drop=True)
print(f'\nControl actions for passing episodes (clipped to window): {len(act)} '
      f'across {act["cluster_id"].nunique()} episodes')

# ════════════════════════════════════════════════════════════════════════════
# 5. Recompute step statistics per (Tag, ActionType)
# ════════════════════════════════════════════════════════════════════════════
act['_step'] = pd.to_numeric(act['Value'], errors='coerce') - pd.to_numeric(act['PrevValue'], errors='coerce')


def _describe(s):
    s = s.dropna()
    if s.empty:
        return [np.nan] * 7
    d = s.describe()
    return [d['mean'], d['std'], d['min'], d['25%'], d['50%'], d['75%'], d['max']]


stat_rows = []
for (tag, atype), g in act.groupby(['Source', 'Description']):
    inc = g.loc[g['action_direction'] == 'increase', '_step']
    dec = g.loc[g['action_direction'] == 'decrease', '_step']
    stat_rows.append([
        tag, atype, len(g),
        int((g['action_direction'] == 'increase').sum()),
        int((g['action_direction'] == 'decrease').sum()),
        int((g['action_direction'] == 'no_change').sum()),
        *_describe(inc), *_describe(dec),
    ])

stat_cols = ['Tag', 'ActionType', 'total_actions', 'increase_count', 'decrease_count', 'no_change_count',
             'inc_step_mean', 'inc_step_std', 'inc_step_min', 'inc_step_25%', 'inc_step_50%', 'inc_step_75%', 'inc_step_max',
             'dec_step_mean', 'dec_step_std', 'dec_step_min', 'dec_step_25%', 'dec_step_50%', 'dec_step_75%', 'dec_step_max']
step_stats = pd.DataFrame(stat_rows, columns=stat_cols).sort_values('total_actions', ascending=False).reset_index(drop=True)
act = act.drop(columns='_step')

# ════════════════════════════════════════════════════════════════════════════
# 5b. Navigable list of the unique passing episodes (one row each + key info)
# ════════════════════════════════════════════════════════════════════════════
act_counts = (act.groupby('cluster_id')
              .agg(n_control_actions=('Description', 'size'),
                   n_OP=('Description', lambda s: int((s == 'OP').sum())),
                   n_SP=('Description', lambda s: int((s == 'SP').sum())),
                   n_MODE=('Description', lambda s: int((s == 'MODE').sum())),
                   n_tags_acted=('Source', 'nunique'),
                   first_action_time=('VT_Start', 'min'),
                   last_action_time=('VT_Start', 'max'))
              .reset_index())

episode_list = passing[['cluster_id', 'cluster_start_time', 'cluster_end_time',
                        'cluster_type', 'cluster_total_alarms', 'cluster_total_duration_min',
                        'window_start', 'window_end',
                        '02FI_1000_max', '03TI_1421_max', '03TI_1901_max']].copy()
episode_list = episode_list.merge(act_counts, on='cluster_id', how='left')
for c in ['n_control_actions', 'n_OP', 'n_SP', 'n_MODE', 'n_tags_acted']:
    episode_list[c] = episode_list[c].fillna(0).astype(int)
episode_list.insert(1, 'episode_folder', episode_list['cluster_id'].map(lambda c: f'episode_{c:04d}'))
episode_list.insert(2, 'episode_path', episode_list['cluster_id'].map(lambda c: f'all_episodes/episode_{c:04d}'))
episode_list = episode_list.rename(columns={'cluster_id': 'episode_id',
                                            '02FI_1000_max': 'FI1000_max',
                                            '03TI_1421_max': 'TI1421_max',
                                            '03TI_1901_max': 'TI1901_max'})
episode_list = episode_list.sort_values('episode_id').reset_index(drop=True)
print(f'Episode list: {len(episode_list)} unique filtered episodes '
      f'({(episode_list["n_control_actions"] == 0).sum()} with no control actions in window)')

# ════════════════════════════════════════════════════════════════════════════
# 6. Write Excel
# ════════════════════════════════════════════════════════════════════════════
with pd.ExcelWriter(OUT_FILE, engine='openpyxl') as xw:
    episode_list.to_excel(xw, sheet_name='episode_list', index=False)
    act.to_excel(xw, sheet_name='control_actions', index=False)
    step_stats.to_excel(xw, sheet_name='step_stats_by_tag_action', index=False)

print(f'\nSaved: {OUT_FILE}')
print(f'Passing episode cluster_ids: {sorted(passing_ids)}')
