---
name: episode-analyzer
description: Comprehensive episode-wise analysis for alarm episodes. Analyzes each alarm episode from transition start through alarm end, calculating rate of change metrics, operating limit deviations, and operator action details. Generates analysis sheets and visualizations for each episode.
metadata:
  author: control-actions-team
  version: "1.0"
  target-alarm: "03LIC_1071"
  created: "2026-01-21"
---

# Episode Analyzer

## Purpose

Performs comprehensive episode-wise analysis for each alarm episode, computing multiple metrics across different time windows within each episode. Generates both analysis sheets (Excel/CSV) and visualizations.

## Episode Time Windows

For each alarm episode, we analyze three key time periods:
1. **Transition Start → Alarm Start**: Early warning period (from `Tag_First_Transition_Start_minutes` to `AlarmStart_rounded_minutes`)
2. **Alarm Start → Alarm End**: Active alarm period
3. **Full Episode**: From `Tag_First_Transition_Start_minutes` to `AlarmEnd_rounded_minutes`

## Metrics Computed

### 1. Rate of Change Metrics (per tag, per episode)
- Mean rate of change (derivative)
- Max/Min rate of change
- Standard deviation of rate of change
- Trend direction (increasing/decreasing/stable)
- Time to peak rate of change

### 2. Operating Limit Deviation (per tag, per episode)
- Deviated from operating limits? (Yes/No)
- Time of first deviation
- Duration of deviation
- Max deviation magnitude (% outside limits)
- Which limit breached (upper/lower/both)

### 3. Operator Action Details (per tag, per episode)
- Number of SP changes
- Number of OP changes
- Direction breakdown (increase/decrease counts for SP and OP)
- Magnitude statistics (mean, median, std dev, min, max)
- Time of first/last action
- Total cumulative change

## Quick Start

```bash
# Run full episode analysis for 2025
python .skills/episode-analyzer/scripts/analyze_episodes.py \
    --start-date 2025-01-01 --end-date 2025-06-30

# Generate episode visualizations
python .skills/episode-analyzer/scripts/generate_episode_plots.py \
    --start-date 2025-01-01 --end-date 2025-06-30 \
    --output-dir RESULTS/episode_plots

# Compute rate of change metrics only
python .skills/episode-analyzer/scripts/compute_rate_of_change.py \
    --start-date 2025-01-01 --end-date 2025-06-30

# Check operating limit deviations
python .skills/episode-analyzer/scripts/check_operating_limits.py \
    --start-date 2025-01-01 --end-date 2025-06-30

# Analyze operator actions per episode
python .skills/episode-analyzer/scripts/analyze_operator_actions.py \
    --start-date 2025-01-01 --end-date 2025-06-30
```

## CLI Options

All scripts support these common options:
- `--start-date YYYY-MM-DD`: Filter data from this date
- `--end-date YYYY-MM-DD`: Filter data until this date
- `--no-trip-filter`: Disable trip period filtering
- `--ground-truth PATH`: Path to ground truth CSV with AlarmStart_rounded column (default: DATA/Updated Ground truth -Adnoc RCA - recent(all_episode_top5_test_validated).csv)
- `--no-ground-truth-filter`: Disable filtering episodes to those in ground truth file
- `--output-dir PATH`: Directory for output files (default: RESULTS/)
- `--output-json`: Also output results in JSON format
- `--ssd-file PATH`: Path to SSD file (default: DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx)
- `--operating-limits PATH`: Path to operating limits file (default: DATA/operating_limits.csv)

### Ground Truth Filtering (Default Enabled)

By default, all scripts now filter episodes to only those present in the ground truth CSV file. This ensures analysis is performed only on validated alarm episodes.

- The ground truth CSV must have an `AlarmStart_rounded` column
- Only episodes whose `AlarmStart_rounded_minutes` matches a value in the ground truth are analyzed
- To disable this filtering and analyze all episodes, use `--no-ground-truth-filter`

## Data Sources

### Input Files
| File | Description |
|------|-------------|
| `DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx` | SSD alarm episodes with transition times |
| `DATA/03LIC_1071_JAN_2026.parquet` | Time series PV/OP data |
| `DATA/df_df_events_1071_export.csv` | Events including CHANGE actions |
| `DATA/operating_limits.csv` | Operating limits per tag |
| `DATA/Final_List_Trip_Duration.csv` | Trip periods to exclude |
| `DATA/Updated Ground truth -Adnoc RCA - recent(all_episode_top5_test_validated).csv` | Ground truth CSV for episode filtering (uses AlarmStart_rounded column) |

### Output Files
| File | Description |
|------|-------------|
| `RESULTS/episode_analysis_summary.xlsx` | Main episode summary with all metrics |
| `RESULTS/episode_rate_of_change.xlsx` | Detailed rate of change per episode/tag |
| `RESULTS/episode_operating_limit_deviations.xlsx` | Operating limit deviations per episode/tag |
| `RESULTS/episode_operator_actions.xlsx` | Operator action details per episode/tag |
| `RESULTS/episode_plots/*.html` | Interactive Plotly visualizations per episode |

## Module Structure

```
.skills/episode-analyzer/
├── SKILL.md                           # This documentation
└── scripts/
    ├── analyze_episodes.py            # Main entry point - runs all analysis
    ├── compute_rate_of_change.py      # Rate of change metrics
    ├── check_operating_limits.py      # Operating limit deviation check
    ├── analyze_operator_actions.py    # Operator action analysis
    └── generate_episode_plots.py      # Visualization generation
```

## Key Concepts

### Episode Definition
An episode is defined by the SSD data with these key timestamps:
- `Tag_First_Transition_Start_minutes`: Earliest transition start across all tags for this episode
- `AlarmStart_rounded_minutes`: When the alarm was triggered
- `AlarmEnd_rounded_minutes`: When the alarm was cleared

### Rate of Change Calculation
For a PV time series within an episode:
```python
# Using rolling derivative
pv_derivative = pv_series.diff() / time_delta_minutes
mean_roc = pv_derivative.mean()
max_roc = pv_derivative.max()
min_roc = pv_derivative.min()
```

### Operating Limit Check
Using `DATA/operating_limits.csv`:
- `LOWER_LIMIT`: Minimum acceptable value
- `UPPER_LIMIT`: Maximum acceptable value
- Deviation = value outside [LOWER_LIMIT, UPPER_LIMIT]

### Operator Action Classification
From CHANGE events:
- **OP changes**: Actions on controller outputs (tags ending in .OP in Source or Description)
- **SP changes**: Actions on setpoints (tags with SP in Description)
- **Direction**: Determined by comparing `Value` vs `PrevValue`
