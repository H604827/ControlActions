---
name: shared
description: Shared utilities for all Control Actions skills. Provides consistent data loading with trip period filtering and date range support. All other skills should import from this module for preprocessing.
metadata:
  author: control-actions-team
  version: "2.0"
  created: "2026-01-20"
  updated: "2026-01-27"
---

# Shared Preprocessing Module

## Purpose

This module provides centralized data loading and preprocessing functions used by all skills. It ensures consistent:
- **Trip period filtering**: Excludes data during plant trips/shutdowns
- **Date range filtering**: Focus analysis on specific time periods
- **Tag name utilities**: Handle naming variations between data sources
- **Episode utilities**: Common functions for SSD, operating limits, and ground truth data

## Module Structure

```
.skills/shared/
├── __init__.py          # Package initialization
├── data_loader.py       # Main data loading functions
├── tag_utils.py         # Tag name utilities
├── episode_utils.py     # Episode analysis utilities (NEW)
└── SKILL.md             # This documentation
```

## How to Use

### Import in Other Skills

```python
import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.data_loader import load_all_data, filter_trip_periods, DataFilterStats
from shared.tag_utils import strings_similar, get_pv_op_pairs, categorize_tags
from shared.episode_utils import (
    load_ssd_data,
    load_operating_limits,
    load_ground_truth_with_fallback,
    get_unique_episodes,
    find_lowest_1071_timestamp,
    compute_percentage_change
)
```

### Load Data with Filtering

```python
# Load both time series and events with full filtering
ts_df, events_df, stats = load_all_data(
    ts_path='DATA/03LIC_1071_JAN_2026.parquet',
    events_path='DATA/df_df_events_1071_export.csv',
    trip_path='DATA/Final_List_Trip_Duration.csv',
    start_date='2025-01-01',
    end_date='2025-06-30',
    filter_trips=True,
    verbose=True
)

# stats is a DataFilterStats object with filtering summary
stats.print_summary()
```

### Load Only Events (for action analysis)

```python
# Pass ts_path=None to skip time series loading
_, events_df, stats = load_all_data(
    ts_path=None,
    events_path='DATA/df_df_events_1071_export.csv',
    start_date='2025-01-01',
    end_date='2025-06-30',
    filter_trips=True
)
```

### Load Only Time Series (for profiling)

```python
# Pass events_path=None to skip events loading
ts_df, _, stats = load_all_data(
    ts_path='DATA/03LIC_1071_JAN_2026.parquet',
    events_path=None,
    start_date='2025-01-01',
    end_date='2025-06-30',
    filter_trips=True
)
```

## Available Functions

### data_loader.py

| Function | Description |
|----------|-------------|
| `load_all_data()` | Main entry point - loads ts and/or events with filtering |
| `load_timeseries_data()` | Load and preprocess time series parquet |
| `load_events_data()` | Load and preprocess events CSV |
| `load_trip_data()` | Load trip duration data |
| `filter_trip_periods()` | Remove rows during trip periods |

### tag_utils.py

| Function | Description |
|----------|-------------|
| `strings_similar()` | Check if two tag names are similar (handles variations) |
| `get_pv_op_pairs()` | Find tags with both PV and OP columns |
| `categorize_tags()` | Categorize columns into controllable/pv-only/op-only |

### episode_utils.py

| Function | Description |
|----------|-------------|
| `load_ssd_data()` | Load and preprocess SSD (Steady State Detection) data |
| `load_operating_limits()` | Load operating limits CSV, indexed by TAG_NAME |
| `load_ground_truth_alarm_starts()` | Load unique alarm start times from ground truth |
| `load_ground_truth_with_fallback()` | Load ground truth with error handling |
| `get_unique_episodes()` | Extract unique episodes from SSD with optional filtering |
| `find_lowest_1071_timestamp()` | Find when target tag PV is at minimum during alarm |
| `compute_percentage_change()` | Calculate percentage change for a tag in a time window |
| `get_tags_in_episode()` | Get tags that transitioned during an episode |
| `get_unique_tags_from_ssd()` | Get unique base tag names from SSD data |

## Episode Utilities Usage

### Load SSD Data

```python
from shared.episode_utils import load_ssd_data, get_unique_episodes

# Load SSD data with date filtering
ssd_df = load_ssd_data(
    ssd_path='DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx',
    start_date='2025-01-01',
    end_date='2025-06-30'
)

# Extract unique episodes
episodes_df = get_unique_episodes(ssd_df)
```

### Load with Ground Truth Filtering

```python
from shared.episode_utils import load_ground_truth_with_fallback, get_unique_episodes

# Load ground truth (returns None if file not found)
gt_alarm_starts = load_ground_truth_with_fallback(
    'DATA/Updated Ground truth -Adnoc RCA - recent(all_episode_top5_test_validated).csv'
)

# Filter episodes to ground truth
episodes_df = get_unique_episodes(ssd_df, ground_truth_alarm_starts=gt_alarm_starts)
```

### Compute Metrics

```python
from shared.episode_utils import find_lowest_1071_timestamp, compute_percentage_change

# Find when target tag is at minimum
lowest_time = find_lowest_1071_timestamp(ts_df, alarm_start, alarm_end)

# Compute percentage change
result = compute_percentage_change(ts_df, transition_start, lowest_time, 'TAG.PV')
print(f"Change: {result['pct_change']:.2f}%")
```

## DataFilterStats Class

The `DataFilterStats` class tracks filtering statistics:

```python
@dataclass
class DataFilterStats:
    # Row counts
    original_ts_rows: int
    original_events_rows: int
    ts_rows_after_date_filter: int
    events_rows_after_date_filter: int
    ts_rows_in_trips: int
    events_rows_in_trips: int
    final_ts_rows: int
    final_events_rows: int
    
    # Data range
    data_start: str
    data_end: str
    data_days: int
    
    # Filter settings
    trips_filtered: bool
    start_date_filter: str
    end_date_filter: str
    
    def to_dict(self) -> dict
    def print_summary(self)
```

## Trip Data Format

The trip duration file (`DATA/Final_List_Trip_Duration.csv`) must have:
- `Trip_Start`: Trip start datetime
- `Trip_End`: Trip end datetime

Example:
```csv
Trip_Start,Trip_End
2024-03-15 08:00:00,2024-03-15 16:00:00
2024-05-20 00:00:00,2024-05-22 12:00:00
```

## Default Paths

```python
DEFAULT_TS_PATH = 'DATA/03LIC_1071_JAN_2026.parquet'
DEFAULT_EVENTS_PATH = 'DATA/df_df_events_1071_export.csv'
DEFAULT_TRIP_PATH = 'DATA/Final_List_Trip_Duration.csv'
DEFAULT_KG_PATH = 'DATA/03LIC1071_PropaneLoop_0426.csv'
```

## Example Output

When `verbose=True`:
```
📊 Data Filtering Summary:
   Original rows: 1,737,586 ts, 1,947,510 events
   Date range filter: 2025-01-01 to 2025-06-30
   After date filter: 242,613 ts, 92,214 events
   Rows removed (in trips): 8,994 ts, 20,302 events
   Final rows for analysis: 233,619 ts, 71,912 events
   Data range: 2025-01-01 to 2025-06-23 (173 days)
```

## Important Notes

1. **No 1.5 hour offset**: The `df_df_events_1071_export.csv` file does NOT need timestamp adjustment. The offset was only required for raw parquet files in `events_1071/` folder.

2. **Trip filtering is recommended**: Always enable trip filtering for analysis to avoid including abnormal plant states.

3. **Date range filtering**: Recommended to focus on recent data (e.g., last 6 months or year) as process dynamics may change over time.
