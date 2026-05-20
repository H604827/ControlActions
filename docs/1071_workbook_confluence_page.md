# 03LIC_1071 Consolidated Alarm, Control Action, and Plant Context Workbook

## Purpose

This page documents the consolidated workbook:

[1071_pvlo_alarms_clustered_with_control_actions_with_plant_context.xlsx](https://honeywellprod-my.sharepoint.com/:x:/g/personal/masum_saikia_honeywell_com/IQCmByP5icNiSKEMruJ0duUrASGJG5f1HydpGsBbtcDIW_w?e=fFkbFU)

Useful links:

- [Excel workbook](https://honeywellprod-my.sharepoint.com/:x:/g/personal/masum_saikia_honeywell_com/IQCmByP5icNiSKEMruJ0duUrASGJG5f1HydpGsBbtcDIW_w?e=fFkbFU)
- [Generation notebook](https://github.com/H604827/ControlActions/blob/main/EXPERIMENTS/control_actions_context.ipynb)

The workbook is the single source for reviewing:

- 03LIC_1071 alarm clusters
- SP and OP changes linked to those alarm clusters
- PV values at the time of action
- ROC-style plant movement across multiple windows
- overlapping alarms and operated tags around the same clusters

The workbook contains three sheets:

1. `alarm_clusters`
2. `control_actions`
3. `overlapping_alarms`

The common key across sheets is `cluster_id`.

## Sheet 2 `control_actions`

If the objective is to review ROC, PV, and SP/OP change mapping, this is the main sheet to start with.

This sheet preserves the original action rows and adds the plant-context fields needed to interpret each action in process context.

At a practical level, this sheet answers the following questions for each action row:

- which tag was changed
- whether the action was on `SP` or `OP`
- what the previous and new values were
- what the signed step change was
- what the target PV and related-tag PV context looked like when the action was taken
- how the plant had moved from deviation start to action time
- how the plant had moved in the short windows just before the action

### Where SP and OP change mapping lives

The SP/OP mapping is carried by these columns:

- `Source`: the tag being changed
- `Description`: identifies the action type, typically `SP` or `OP`
- `VT_Start`: the original action timestamp
- `PrevValue`: the previous setting
- `Value`: the new setting
- `Step`: the raw signed step change

For same-minute repeated actions, the workbook also provides merged-action fields:

- `minute_bucket`
- `merged_group_id`
- `merged_group_role`
- `merged_action_timestamp`
- `merged_prev_value`
- `merged_value`
- `merged_step`
- `merged_action_direction`
- `merged_num_actions`

This gives both views:

- the original row-level event history
- a consolidated same-minute action mapping for cleaner analysis

Definition of `merged_action_timestamp`:

- rows are first grouped by `cluster_id`, `Source`, `Description`, and `minute_bucket`
- within each such group, rows are sorted by `VT_Start`
- `merged_action_timestamp` is the first `VT_Start` in that sorted same-minute group

So this field is the representative start time of the merged SP or OP action block, not an average timestamp and not the last action time in that minute.

### Where PV and ROC-style context lives

The action-time PV and ROC-style fields are also on the `control_actions` sheet.

Examples:

- `merged_ctx_03LIC_1071_pv_at_action`
- `merged_ctx_alarm_proximity`
- `merged_ctx_time_progress_ratio`
- `merged_ctx_*_norm_pos`
- `merged_ctx_*_episode_norm_roc`
- `merged_ctx_*_local_3m_delta_norm`
- `merged_ctx_*_local_5m_delta_norm`

These fields make it possible to see, for each action:

- the PV value when the action was taken
- the normalized plant state across different tags at action time
- the direction and magnitude of movement from deviation start to action time
- the short-window movement immediately before the action

## Plant Context: Definition and Formulas

The plant context is defined around the merged action timestamp rather than only around the raw action row timestamp. This provides one consistent context snapshot for each same-minute action group.

For reliable rendering in Confluence and markdown viewers, the formulas below are written in plain-text notation.

### Time definitions

Let:

- `t_a` = merged action timestamp, i.e. the earliest original `VT_Start` inside the merged same-minute action group
- `t_d` = deviation start timestamp from SSD
- `t_a_minus_3` = `t_a - 3 minutes`
- `t_a_minus_5` = `t_a - 5 minutes`

For any PV tag `x`:

- `PV_x(t)` = historian PV value for tag `x` at time `t`, using as-of lookup
- `LL_x` = lower operating limit for tag `x`
- `UL_x` = upper operating limit for tag `x`
- `Range_x = UL_x - LL_x`

The historian lookup uses an as-of snapshot:

- first it takes the latest available value at or before the requested timestamp
- if no earlier value exists, it falls forward to the next available value

### 1. Raw action step

For the original action row:

`Step = Value - PrevValue`

### 2. Same-minute action merge logic

Rows are grouped using:

- `cluster_id`
- `Source`
- `Description`
- `minute_bucket`

where:

`minute_bucket = floor_to_minute(VT_Start)`

For each merged group:

`merged_step = merged_value - merged_prev_value`

where:

- `merged_prev_value` is the first numeric value in the sorted group
- `merged_value` is the last numeric value in the sorted group
- `merged_action_timestamp` is the first timestamp in the sorted group

This creates one representative action view for repeated SP/OP changes in the same minute.

### 3. SSD timing linkage

Each alarm cluster is matched to SSD using normalized start and end times.

The control cluster timestamps are normalized as:

`cluster_start_floor = floor_to_minute(round_to_second(cluster_start))`

`cluster_end_floor = floor_to_minute(round_to_second(cluster_end))`

If a cluster matches an SSD window, then:

`merged_minutes_from_deviation = (t_a - t_d) / 60 seconds`

### 4. PV at action time

For the target tag:

`merged_ctx_03LIC_1071_pv_at_action = PV_03LIC_1071(t_a)`

This is the direct PV value when the merged action starts.

### 5. Normalized PV position within operating range

For any valid PV tag `x`:

`merged_ctx_{x}_norm_pos = (PV_x(t_a) - LL_x) / Range_x`

Interpretation:

- near 0 means closer to the lower operating limit
- near 1 means closer to the upper operating limit

### 6. Episode-level ROC-style movement from deviation start to action

For any valid PV tag `x`:

`merged_ctx_{x}_episode_norm_roc = (PV_x(t_a) - PV_x(t_d)) / Range_x`

This is the workbook's main episode-level ROC-style feature. It is a normalized change from deviation start to action time.

Important note:

- this is a window-based change metric, not a continuous derivative
- it captures direction and magnitude of movement over the episode window

### 7. Local 3-minute ROC-style movement before action

For any valid PV tag `x`:

`merged_ctx_{x}_local_3m_delta_norm = (PV_x(t_a) - PV_x(t_a_minus_3)) / Range_x`

This captures short-window movement immediately before the action.

### 8. Local 5-minute ROC-style movement before action

For any valid PV tag `x`:

`merged_ctx_{x}_local_5m_delta_norm = (PV_x(t_a) - PV_x(t_a_minus_5)) / Range_x`

This captures a slightly broader short-window movement immediately before the action.

### 9. Alarm proximity for the target tag

Let `Threshold = 28.75` for `03LIC_1071.PV`.

`merged_ctx_alarm_proximity = (PV_03LIC_1071(t_a) - Threshold) / (UL_03LIC_1071 - Threshold)`

Interpretation:

- 0 means the target PV is exactly at the alarm threshold
- positive values mean the PV is above the threshold
- negative values mean the PV is below the threshold

### 10. Time progress ratio within the cluster lifecycle

Let `MedianClusterDuration` be the median duration across clusters.

`merged_ctx_time_progress_ratio = merged_minutes_from_deviation / MedianClusterDuration`

Interpretation:

- values below 1 indicate earlier actions
- values above 1 indicate later actions

## How the Workbook Is Built

The workbook combines four main sources:

### 1. Events data

Files used:

- `DATA/trip_filtered_events.csv`
- `DATA/trip_filtered_events_dedup.csv`

Used for:

- identifying 03LIC_1071 alarm episodes
- building the alarm clusters
- identifying operator actions
- restoring reliable `PrevValue` and `Value` fields for the control action rows

### 2. SSD data

File used:

- `DATA/SSD_1071_ControlActions_24April2026.csv`

Used for:

- identifying deviation start time for a matched alarm cluster
- anchoring action timing relative to process deviation

### 3. Historian minute-wise data

File used:

- `DATA/03LIC_1071_JAN_2026_filtered.parquet`

Used for:

- reading PV values at action time
- reading PV values at deviation start
- reading PV values 3 minutes before action
- reading PV values 5 minutes before action

### 4. Operating limits

File used:

- `DATA/operating_limits.csv`

Used for:

- converting raw PV values into normalized plant-context features
- making multi-tag comparisons easier across different engineering ranges

## Other Sheets in the Workbook

## Sheet 1. `alarm_clusters`

This sheet is the master cluster definition sheet for 03LIC_1071 alarms.

The starting point is the events data for `03LIC_1071`.

Alarm logic:

- alarm start: `ConditionName = PVLO` and `Action` is blank or null
- alarm end: `ConditionName = PVLO` and `Action = OK`

These start and end pairs define raw alarm episodes.

The clustering rule is:

- if two consecutive alarms are less than 30 minutes apart, they belong to the same alarm cluster

This sheet therefore compiles the individual 03LIC_1071 alarms into larger operating episodes.

Key fields include:

- `episode_num`
- `alarm_start`
- `alarm_end`
- `duration_minutes`
- `gap_to_next_minutes`
- `cluster_id`
- `cluster_total_alarms`
- `cluster_start_time`
- `cluster_end_time`
- `cluster_total_duration_min`
- `gap_to_next_cluster_min`
- `cluster_type`

Use this sheet when the question is:

- what are the 03LIC_1071 alarm clusters
- how many alarms were rolled into each cluster
- what is the time span of each cluster

## Sheet 3. `overlapping_alarms`

This sheet provides cluster-level overlap context.

For each 03LIC_1071 alarm cluster, it contains:

- which other tags had overlapping alarms with 03LIC_1071
- how many overlapping alarms were present
- filtered overlap counts
- how many tags were operated during the cluster
- filtered operated-tag counts
- filtered tag lists for both overlapping and operated tags

Key fields include:

- `episode_idx`
- `cluster_id`
- `alarm_start`
- `alarm_end`
- `duration_min`
- `cluster_total_alarms`
- `cluster_type`
- `num_overlapping_alarms`
- `num_overlapping_filtered`
- `num_operated_tags`
- `num_operated_tags_filtered`
- `filtered_tags_str`
- `operated_tags_filtered_str`

The filtered counts and filtered tag strings are useful when looking at the overlap context after KG-path or analysis filtering.

Use this sheet when the question is:

- what else was alarming during a given 03LIC_1071 cluster
- how large the concurrent alarm context was
- which filtered tags and operated tags were relevant to that cluster

## Generation Logic

The workbook logic is implemented in:

- [control_actions_context.ipynb](https://github.com/H604827/ControlActions/blob/main/EXPERIMENTS/control_actions_context.ipynb)

This notebook:

- loads the original workbook
- enriches the `control_actions` sheet
- maps SSD timing
- computes historian-based plant context
- preserves the original workbook sheets while saving the consolidated workbook
