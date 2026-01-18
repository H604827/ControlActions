---
name: process-data-explorer
description: Analyze industrial process data (PV/OP time series, events, alarms) for the Control Actions project. Use when exploring parquet/CSV data, understanding tag relationships, profiling data quality, identifying PV-OP pairs, or mapping event sources to time series columns.
metadata:
  author: control-actions-team
  version: "1.1"
  target-alarm: "03LIC_1071"
  last-run: "2026-01-18"
---

# Process Data Explorer

## When to Use This Skill

Use this skill when you need to:
- Profile and understand the structure of PV/OP time series data
- Explore events/actions data and understand its relationship to time series
- Identify which tags have both `.PV` and `.OP` columns
- Map tag names between different data sources (knowledge graph, events, time series)
- Generate data quality reports

## Latest Run Results (Jan 2026)

The scripts have been run and outputs saved to:
- `RESULTS/timeseries_profile.json` - Full time series profile
- `RESULTS/data_relationships.json` - Tag mapping and relationships

**Key findings:**
- 15 controllable tags (have both PV and OP)
- 13 PV-only tags (monitoring only)
- 1,737,586 rows of minute-wise data (2022-01-03 to 2025-06-23)
- 41 time gaps >5 minutes (largest: 446 hours)
- 1,861 PVLO alarm episodes for target tag

## Data Sources Overview

### 1. PV/OP Time Series (`DATA/03LIC_1071_JAN_2026.parquet`)
- **Columns**: End with `.PV` (Process Variable) or `.OP` (Output)
- **Special Columns**: `AlarmStatus` (ON/OFF), `AlarmType` (PVLO/blank) - these are object type, not numeric
- **Index**: TimeStamp (minute-wise readings)
- **Range**: 2022-01-03 to 2025-06-23 (~3.5 years)

### 2. Events Data (`DATA/df_df_events_1071_export.csv`)
- **Key Columns**: `Source`, `VT_Start`, `ConditionName`, `Action`, `Value`, `PrevValue`
- **Event Types**: `CHANGE` (operator actions), `PVLO`/`PVHI` (alarms)

### 3. Related Tags (`DATA/03LIC1071_PropaneLoop_0426.csv`)
- **Column**: `tagName` - tags from knowledge graph related to target alarm

## Step-by-Step Instructions

### Step 1: Load and Profile Time Series Data

```python
import pandas as pd

# Load PV/OP data
op_pv_data_df = pd.read_parquet('DATA/03LIC_1071_JAN_2026.parquet')
op_pv_data_df.set_index('TimeStamp', inplace=True)
op_pv_data_df.sort_index(inplace=True)

# Identify PV and OP columns
cols = op_pv_data_df.columns
op_tags = {col.replace('.OP', '') for col in cols if col.endswith('.OP')}
pv_tags = {col.replace('.PV', '') for col in cols if col.endswith('.PV')}

# Find tags with BOTH PV and OP (controllable tags)
controllable_tags = op_tags & pv_tags
print(f"Tags with both PV and OP: {len(controllable_tags)}")
print(f"Tags with only OP: {op_tags - pv_tags}")
print(f"Tags with only PV: {pv_tags - op_tags}")
```

### Step 2: Profile Data Quality

Run the profiling script:
```bash
python .skills/process-data-explorer/scripts/profile_timeseries.py
```

Or manually:
```python
# Check time range
print(f"Time range: {op_pv_data_df.index.min()} to {op_pv_data_df.index.max()}")

# Check for gaps in time series
time_diff = op_pv_data_df.index.to_series().diff()
gaps = time_diff[time_diff > pd.Timedelta(minutes=5)]
print(f"Number of gaps > 5 minutes: {len(gaps)}")

# Missing values per column
missing = op_pv_data_df.isnull().sum()
print(f"Columns with missing values: {missing[missing > 0]}")
```

### Step 3: Load and Profile Events Data

```python
# Load events data
events_df = pd.read_csv("DATA/df_df_events_1071_export.csv", low_memory=False)
events_df['VT_Start'] = pd.to_datetime(events_df['VT_Start'])
events_df = events_df.sort_values('VT_Start')

# Profile event types
print("Event types (ConditionName):")
print(events_df['ConditionName'].value_counts())

# CHANGE events = operator/automated actions
change_events = events_df[events_df['ConditionName'] == 'CHANGE']
print(f"\nCHANGE events: {len(change_events)}")
print(f"Unique sources with CHANGE: {change_events['Source'].nunique()}")
```

### Step 4: Map Tags Between Data Sources

Use the `strings_similar()` function to match tags across sources:

```python
def strings_similar(s1, s2):
    """Match tags with slight naming variations."""
    s1 = str(s1).strip().replace(' ', '').upper()
    s2 = str(s2).strip().replace(' ', '').upper()
    
    if len(s1) < 3 or len(s2) < 3:
        return False
    if s1[:3] != s2[:3]:
        return False
    
    if '_' not in s1 or '_' not in s2:
        return False
    
    s1_after = s1.split('_', 1)[1]
    s2_after = s2.split('_', 1)[1]
    
    if s1_after == s2_after:
        return True
    
    shorter = s1_after if len(s1_after) <= len(s2_after) else s2_after
    longer = s2_after if len(s1_after) <= len(s2_after) else s1_after
    
    for i in range(len(shorter) - 3):
        if shorter[i:i+4] in longer:
            return True
    return False
```

### Step 5: Generate Relationship Summary

Run the relationship mapping script:
```bash
python .skills/process-data-explorer/scripts/map_data_relationships.py
```

## Key Outputs

After running this skill, you should have:
1. **Tag inventory**: List of all PV/OP tag pairs
2. **Data quality report**: Missing values, gaps, anomalies
3. **Tag mapping**: How tags in events relate to time series columns
4. **Time range alignment**: Verify events and time series cover same period

## Common Issues

- **Tag name mismatches**: Use `strings_similar()` for fuzzy matching
- **Timezone issues**: Ensure all timestamps are in same timezone
- **Category filtering**: Use `Category == 1` for filtering relevant events
