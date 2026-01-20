---
name: operator-action-learner
description: Analyze historical operator actions (CHANGE events) to learn intervention patterns, action magnitudes, directions, and sequencing for the Control Actions project. Use when understanding what actions operators take during alarms, learning action magnitudes, or building training data for ML models.
metadata:
  author: control-actions-team
  version: "2.0"
  target-alarm: "03LIC_1071"
---

# Operator Action Learner

## Version 2.0 Updates

- **Trip Period Filtering**: Automatically excludes data during plant trips
- **Date Range Filtering**: Focus analysis on specific time periods
- **Shared Preprocessing**: Uses centralized `shared/data_loader.py` for consistent data handling

## When to Use This Skill

Use this skill when you need to:
- Understand what actions operators took during alarm episodes
- Extract action magnitudes (how much did they change OP values?)
- Determine action direction (increase or decrease?)
- Learn sequencing patterns (which tags are operated first?)
- Build training datasets for ML models
- Correlate action patterns with alarm resolution success

## Quick Start

```bash
# Analyze action magnitudes for 2025 data (with trip filtering)
python .skills/operator-action-learner/scripts/analyze_action_magnitudes.py \
    --start-date 2025-01-01 --end-date 2025-06-30

# Extract action sequences during alarm episodes
python .skills/operator-action-learner/scripts/extract_action_sequences.py \
    --start-date 2025-01-01 --end-date 2025-06-30

# Build ML training features (filtered)
python .skills/operator-action-learner/scripts/build_training_features.py \
    --start-date 2025-01-01 --end-date 2025-06-30 \
    --output-file RESULTS/training_features_2025.csv
```

## CLI Options

All scripts support these common options:
- `--start-date YYYY-MM-DD`: Filter data from this date
- `--end-date YYYY-MM-DD`: Filter data until this date
- `--trip-file PATH`: Path to trip duration file (default: DATA/Final_List_Trip_Duration.csv)
- `--no-trip-filter`: Disable trip period filtering
- `--recent`: Analyze only recent 6 months
- `--last-year`: Analyze only last year of data
- `--output-json`: Output results in JSON format

## Core Concept

**Hypothesis**: Operators decide actions based on:
- Current PV value and its distance from alarm threshold
- Rate of change (derivative) of PV
- Values of related tags
- Historical experience

By analyzing historical actions, we can learn:
1. **WHAT MAGNITUDE**: How much step change to apply
2. **WHAT DIRECTION**: Increase or decrease OP
3. **WHICH TAGS**: Which tags to operate on
4. **IN WHAT ORDER**: Sequencing of actions

## Data Sources

### CHANGE Events
From `DATA/df_df_events_1071_export.csv`:
- `Source`: Tag that was operated on
- `VT_Start`: When action was taken
- `Value`: New OP value after action
- `PrevValue`: OP value before action
- `ConditionName`: Must be 'CHANGE'

### Alarm Episodes
From alarm Start/End pairs:
- Define time windows for analysis
- Link actions to specific alarm events

## Step-by-Step Instructions

### Step 1: Extract Actions During Alarm Episodes

```python
import pandas as pd
import numpy as np

# Load data
events_df = pd.read_csv('DATA/df_df_events_1071_export.csv', low_memory=False)
events_df['VT_Start'] = pd.to_datetime(events_df['VT_Start'])

# Get alarm episodes (from previous analysis)
# alarm_episodes = [[start1, end1], [start2, end2], ...]

# Get CHANGE events
change_events = events_df[events_df['ConditionName'] == 'CHANGE'].copy()

# Calculate action magnitude
change_events['magnitude'] = pd.to_numeric(change_events['Value'], errors='coerce') - \
                              pd.to_numeric(change_events['PrevValue'], errors='coerce')

# Filter actions during alarm episodes
def get_actions_during_alarm(alarm_start, alarm_end, change_df):
    mask = (change_df['VT_Start'] >= alarm_start) & \
           (change_df['VT_Start'] <= alarm_end)
    return change_df[mask].copy()
```

### Step 2: Analyze Action Magnitudes

Run the magnitude analysis script:
```bash
python .skills/operator-action-learner/scripts/analyze_action_magnitudes.py
```

Key metrics to extract:
- Mean/median magnitude per tag
- Standard deviation (consistency of actions)
- Distribution of positive vs negative changes

### Step 3: Learn Action Direction Rules

```python
def analyze_action_direction(actions_df, ts_df, target_pv_col='03LIC_1071.PV'):
    """
    Analyze what determines whether operator increases or decreases OP.
    """
    results = []
    
    for _, action in actions_df.iterrows():
        action_time = action['VT_Start']
        tag = action['Source']
        magnitude = action['magnitude']
        
        if pd.isna(magnitude):
            continue
        
        # Get PV state at action time
        try:
            pv_at_action = ts_df.loc[:action_time, target_pv_col].iloc[-1]
            
            # Calculate rate of change (last 5 minutes)
            recent_pv = ts_df.loc[:action_time, target_pv_col].tail(5)
            pv_rate = (recent_pv.iloc[-1] - recent_pv.iloc[0]) / 5 if len(recent_pv) >= 2 else 0
            
            results.append({
                'tag': tag,
                'action_time': action_time,
                'magnitude': magnitude,
                'direction': 'increase' if magnitude > 0 else 'decrease',
                'target_pv_at_action': pv_at_action,
                'target_pv_rate': pv_rate,
                'distance_to_alarm': pv_at_action - 28.75  # For PVLO alarm
            })
        except Exception:
            continue
    
    return pd.DataFrame(results)
```

### Step 4: Extract Sequencing Patterns

```python
def extract_action_sequences(alarm_episodes, change_events):
    """
    Extract sequences of actions during each alarm episode.
    """
    sequences = []
    
    for i, (start, end) in enumerate(alarm_episodes):
        episode_actions = change_events[
            (change_events['VT_Start'] >= start) &
            (change_events['VT_Start'] <= end)
        ].sort_values('VT_Start')
        
        if len(episode_actions) == 0:
            continue
        
        sequence = {
            'episode_id': i,
            'alarm_start': start,
            'alarm_end': end,
            'duration_minutes': (end - start).total_seconds() / 60,
            'action_count': len(episode_actions),
            'action_sequence': episode_actions[['Source', 'VT_Start', 'magnitude']].to_dict('records'),
            'first_action_tag': episode_actions.iloc[0]['Source'],
            'first_action_delay': (episode_actions.iloc[0]['VT_Start'] - start).total_seconds() / 60,
            'tags_operated': episode_actions['Source'].unique().tolist()
        }
        
        sequences.append(sequence)
    
    return sequences
```

### Step 5: Build Feature Set for ML

Run the feature extraction script:
```bash
python .skills/operator-action-learner/scripts/build_training_features.py
```

Features to include:
1. **Target tag features**: PV value, rate of change, distance to threshold
2. **Related tag features**: PV values of knowledge graph tags
3. **Historical features**: Recent actions, time since last action
4. **Temporal features**: Time of day, day of week

## Expected Patterns to Find

### Magnitude Patterns
- Typical step sizes: 1-5% of operating range
- Larger steps when further from threshold
- Smaller steps when close to threshold (fine-tuning)

### Direction Patterns  
- For PVLO alarms (level too low):
  - Increase inflow OP
  - Decrease outflow OP
- Direction depends on tag's relationship to target

### Sequencing Patterns
- Primary tags operated first
- Secondary adjustments follow
- Time gaps between actions (wait for response)

## Key Outputs

After running this skill:
1. **Action magnitude statistics** per tag
2. **Direction correlation** with PV state
3. **Sequence patterns** across episodes
4. **Feature matrix** for ML training
5. **Success correlation** (which patterns resolved alarms faster)

## Analysis Scripts

| Script | Purpose |
|--------|---------|
| `analyze_action_magnitudes.py` | Statistics on action sizes |
| `analyze_action_directions.py` | Correlate direction with plant state |
| `extract_action_sequences.py` | Sequence mining per episode |
| `build_training_features.py` | Create ML-ready dataset |

## Common Findings

- Most actions are small adjustments (< 5% change)
- First action often on the tag closest to the alarm
- Successful resolutions show quicker first response
- Oscillation indicates overcorrection (too large steps)
