# Copilot Instructions for Control Actions Project

## Project Overview

This is an **experimental research project** for **UC3: Autonomous Alarm and Anomaly Resolution** as part of the ADNOC GAS Plant 3 Train 1 pilot. The outcomes of this project (algorithms, ML models, or other solutions) will be used in production.

### Plant Context
- **Plant**: AGP BUHASA - Propane Refrigeration & Extraction Plant (Plant 3, Train 1)
- **Purpose**: Takes compressed gases from Gas Compression Plant (Plant 2), cools and stabilizes them to produce liquid suitable for pumping to Ruwais plant
- **Refrigeration Section**: Provides refrigeration duty necessary for cooling

### Business Objective
Build an automated system that:
1. Takes remedial control actions to prevent alarms (early detection is handled by upstream Anomaly Detection module)
2. Resolves alarms if they do occur
3. Operates the plant autonomously with minimal human intervention

**Note**: This project focuses specifically on the **Control Actions** component. Early alarm detection is handled by the upstream Anomaly Detection module, which is outside the scope of this project.

---

## Current Focus: Target Alarm `03LIC_1071`

### Target Tag Details
- **Tag**: `03LIC_1071.PV`
- **Description**: 3E107 LEVEL (Level Indicator Controller)
- **Alarm Type**: PVLO (Process Variable Low) - most frequent and common alarm type
- **Alarm Threshold**: 28.75 (low limit)
- **Priority**: Low

### Process Response Dynamics (Key Research Area)
- **Critical Question**: For each PV/OP tag pair, how long does it take for an OP change to affect the PV?
- **Observation**: From sample data, PV responds to OP changes relatively quickly (within minutes), but this needs to be systematically quantified
- **Approach Needed**: Develop a method to calculate the response lag/delay for each tag pair
- This response time is essential for:
  - Knowing when to expect results after taking an action
  - Deciding when to take the next action
  - Avoiding overcorrection by acting too quickly

### Alarm Identification Logic
- Alarm **Start**: When `Action` is null/blank and `ConditionName` = 'PVLO'
- Alarm **End**: When `Action` = 'OK' following a Start event
- These consecutive pairs define alarm episodes with start and end timestamps

---

## System Architecture (Upstream Components)

The Control Actions module is part of a larger pipeline:

1. **Anomaly Detection Model**: Runs when any PV tag value is out of its operating limit range
2. **RCA (Root Cause Analysis) Algorithm**: If anomaly is flagged, identifies root cause and provides 3-5 candidate tags
3. **Control Actions Module** (THIS PROJECT): Given candidate tags, determines and executes appropriate actions

---

## Data Sources

### 1. PV/OP Time Series Data
- **Location**: `DATA/03LIC_1071_JAN_2026.parquet` (primary), `DATA/op_pv_data.parquet` (alternative)
- **Frequency**: Minute-wise readings
- **Time Range**: 2022 to mid-2025
- **Structure**: 
  - Columns ending with `.PV` = Process Variable values (sensor readings)
  - Columns ending with `.OP` = Output/Operator values (control outputs)
- **Index**: TimeStamp (set with `op_pv_data_df.set_index('TimeStamp', inplace=True)`)
- **Loading Pattern**:
```python
op_pv_data_df = pd.read_parquet('DATA/03LIC_1071_JAN_2026.parquet')
op_pv_data_df.set_index('TimeStamp', inplace=True)
op_pv_data_df.sort_index(inplace=True)
```

### 2. Events/Actions Data
- **Location**: `DATA/df_df_events_1071_export.csv`
- **Content**: Operator actions and alarm events
- **Key Columns**:
  - `Source`: Tag name (e.g., '03LIC_1071')
  - `VT_Start`: Event timestamp (convert with `pd.to_datetime()`)
  - `ConditionName`: Event type ('CHANGE', 'PVLO', 'PVHI', etc.)
  - `Action`: Event action ('OK' for alarm clear, blank/null for alarm start)
  - `Value`, `PrevValue`: Current and previous values for CHANGE events
  - `Description`: Event description
  - `Category`: Event category (use Category=1 for filtering)
- **Loading Pattern**:
```python
combined_pv_events_df = pd.read_csv("DATA/df_df_events_1071_export.csv", low_memory=False)
combined_pv_events_df['VT_Start'] = pd.to_datetime(combined_pv_events_df['VT_Start'])
combined_pv_events_df = combined_pv_events_df.sort_values('VT_Start')
```

### 3. Related Tags (Knowledge Graph)
- **Location**: `DATA/03LIC1071_PropaneLoop_0426.csv`
- **Content**: Predefined set of related tags from knowledge graph
- **Column**: `tagName` contains the related tag names
- **Usage**: These are potential tags that influence or are influenced by the target tag

### 4. Steady State Detection (SSD) Data
- **Location**: `DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx`
- **Content**: Identifies when tags are out of steady state during alarm episodes
- **Key Columns**:
  - `TagName`: The tag being analyzed
  - `AlarmStart_rounded_minutes`, `AlarmEnd_rounded_minutes`: Alarm episode boundaries

### 5. Output Grids (Generated)
- **Location**: `RESULTS/`
- `alarm_episode_tag_grid.xlsx`: Which tags had CHANGE events during each alarm episode
- `ssd_alarm_episode_tag_grid.xlsx`: Which tags were out of steady state during each alarm episode

---

## Core Problem Statement

### The Challenge
Given:
- A target tag (`03LIC_1071.PV`) approaching or in alarm state
- A list of candidate tags to operate on (from RCA or knowledge graph)
- Current state of plant (all PV tag values)
- Historical operator actions

Determine:
1. **WHICH** tags to operate on (sequence/priority)
2. **WHEN** to take each action
3. **WHAT DIRECTION** - increase or decrease the OP value
4. **WHAT MAGNITUDE** - how much step change to apply

### Hypothesis
Operators decide actions based on:
- PV values of target tag and related tags
- Rate of change of these PV values
- Current plant state

### Action-Observation Loop
1. Analyze current state
2. Decide on an action (tag, direction, magnitude)
3. Apply the action (change OP value)
4. **Observe** the process behavior response
5. Decide on next action
6. Repeat until target tag PV returns to steady state

---

## Key Definitions

### Steady State
- Defined by a specific range based on **IQR (Interquartile Range)** of a particular time window
- When a tag's value is within this range and stable, it is in steady state
- Goal is to bring `03LIC_1071.PV` back to steady state through actions

### CHANGE Events
- Records of when OP/SP values were modified
- Contains `Value` (new) and `PrevValue` (old)
- Indicates operator interventions on specific tags

### Alarm Episode
- Period between alarm Start and End
- Identified by consecutive PVLO events where Action goes from blank → 'OK'

---

## Current State of Analysis

### Completed Work (in `1071_control_actions.ipynb`)
1. ✅ Data loading and preprocessing
2. ✅ Alarm episode extraction from event data
3. ✅ Matching related tags from knowledge graph to event sources
4. ✅ Grid creation: Which tags were operated during each alarm episode
5. ✅ Grid creation: Which tags were out of steady state during each alarm episode
6. ✅ Visualization of PV trends with alarm periods and change events

### Open Research Questions
1. How to model the relationship between OP changes and PV responses?
2. What features best predict the required action magnitude?
3. How to determine action direction (increase vs decrease)?
4. What is the optimal sequencing of actions on multiple tags?
5. How to account for process dynamics (delays, interactions)?

---

## Technical Guidelines

### Data Processing
- Always handle timezone-aware timestamps properly
- PV data index should be sorted chronologically
- Use `.copy()` when filtering DataFrames to avoid SettingWithCopyWarning

### Tag Naming Conventions
- Format: `XXYYYY_ZZZZ` (e.g., `03LIC_1071`)
  - `XX`: Plant/area code
  - `YYYY`: Equipment/instrument type (LIC = Level Indicator Controller)
  - `ZZZZ`: Tag number
- `.PV` suffix = Process Variable
- `.OP` suffix = Output/Operator value

### Extracting OP vs PV Tags
```python
cols = op_pv_data_df.columns
op_tags = {col.replace('.OP', '') for col in cols if col.endswith('.OP')}
pv_tags = {col.replace('.PV', '') for col in cols if col.endswith('.PV')}
```

### String Matching for Tags
Tags in different data sources may have slight variations. Use the `strings_similar()` function defined in `1071_control_actions.ipynb`:
- First 3 characters must match exactly
- Characters after underscore must match or have 4+ character common substring

### Alarm Episode Extraction Pattern
```python
# Filter for target alarm events
df_filtered = combined_pv_events_df[
    (combined_pv_events_df['Source'] == '03LIC_1071') &
    (combined_pv_events_df['ConditionName'] == 'PVLO') &
    (combined_pv_events_df['Category'] == 1)
].copy()

# Identify Start (Action is null/blank) followed by End (Action == 'OK')
is_null_or_blank = df_filtered['Action'].isna() | (df_filtered['Action'] == '')
is_ok = df_filtered['Action'] == 'OK'
```

### Visualization with Plotly
Use `plotly.graph_objects` for interactive visualizations with:
- PV traces as line plots
- Alarm periods as `vrect` shaded regions
- CHANGE events as scatter markers with hover data

---

## Success Criteria

### For Alarm Prevention
- Detect early warning signs before alarm threshold is breached
- Take proactive actions that keep PV within safe range
- Measure: Reduction in alarm occurrences

### For Alarm Resolution
- When alarm occurs, take actions that resolve it
- Minimize time in alarm state
- Avoid overcorrection (causing opposite alarm)
- Measure: Reduction in alarm duration

---

## Approach Considerations

### Current Focus
- **Primary**: `03LIC_1071` PVLO alarm (threshold 28.75)
- **Goal**: Develop approach that can be generalized to other alarms (see list of 20 target alarms in UC3)
- **Other alarms include**: `03LIC_1016`, `03TI_1901`, `03LIC_1097`, `03LIC_1094`, `03LIC_1608`, etc.

### Potential Modeling Approaches
1. **Supervised Learning**: Learn action magnitude from historical operator actions
2. **Reinforcement Learning**: Learn optimal policy through simulation
3. **Rule-based/Heuristic**: Derive rules from process knowledge and data analysis
4. **Model Predictive Control**: Build process models and optimize actions
5. **Hybrid**: Combine multiple approaches

### Key Features to Consider
- Current PV value of target tag
- Distance from alarm threshold
- Rate of change (derivative) of PV
- PV values of related tags
- Current OP values
- Historical action patterns for similar situations

### Constraints to Consider
- Process dynamics (actions take time to have effect)
- Interactions between tags (changing one affects others)
- Safety limits (physical constraints on equipment)
- Avoid oscillation/hunting behavior

---

## Available Skills

The `.skills/` directory contains reusable analysis tools for this project. Each skill includes a `SKILL.md` with instructions and executable Python scripts.

### 1. process-data-explorer
**Purpose**: Understand and profile the available data sources.
**When to use**: Starting analysis, understanding data structure, mapping relationships between datasets.

```bash
# Profile time series data
python .skills/process-data-explorer/scripts/profile_timeseries.py --file DATA/03LIC_1071_JAN_2026.parquet

# Map relationships between data sources
python .skills/process-data-explorer/scripts/map_data_relationships.py
```

### 2. response-dynamics-estimator
**Purpose**: Estimate how long OP changes take to affect PV values.
**When to use**: Understanding process dynamics, calculating response lags, time constant estimation.

```bash
# Estimate dynamics for all OP/PV pairs
python .skills/response-dynamics-estimator/scripts/estimate_dynamics.py \
    --ts-file DATA/03LIC_1071_JAN_2026.parquet \
    --events-file DATA/df_df_events_1071_export.csv

# Visualize dynamics results
python .skills/response-dynamics-estimator/scripts/visualize_dynamics.py
```

### 3. operator-action-learner
**Purpose**: Analyze historical operator actions to learn intervention patterns.
**When to use**: Learning action magnitudes, directions, sequencing patterns, building ML training data.

```bash
# Analyze action magnitudes
python .skills/operator-action-learner/scripts/analyze_action_magnitudes.py \
    --events-file DATA/df_df_events_1071_export.csv

# Extract action sequences during alarm episodes
python .skills/operator-action-learner/scripts/extract_action_sequences.py \
    --events-file DATA/df_df_events_1071_export.csv \
    --target-tag 03LIC_1071 --alarm-type PVLO

# Build training features for ML
python .skills/operator-action-learner/scripts/build_training_features.py \
    --events-file DATA/df_df_events_1071_export.csv \
    --ts-file DATA/03LIC_1071_JAN_2026.parquet \
    --output-file RESULTS/training_features.csv
```

All scripts support `--output-json` for machine-readable output.

---

## File Structure

```
ControlActions/
├── .github/
│   └── copilot-instructions.md    # This file
├── .skills/
│   ├── process-data-explorer/     # Data profiling skill
│   ├── response-dynamics-estimator/ # Dynamics analysis skill
│   └── operator-action-learner/   # Action learning skill
├── DATA/
│   ├── 03LIC1071_PropaneLoop_0426.csv           # Related tags from KG
│   ├── 03LIC_1071_JAN_2026.parquet              # PV/OP time series
│   ├── df_df_events_1071_export.csv             # Events and actions
│   ├── SSD_1071_SSD_output_1071_7Jan2026.xlsx   # Steady state detection
│   └── events_1071/                              # Event subfolders
├── RESULTS/
│   ├── alarm_episode_tag_grid.xlsx              # Actions per alarm episode
│   └── ssd_alarm_episode_tag_grid.xlsx          # SSD per alarm episode
├── 1071_control_actions.ipynb                    # Main analysis notebook
└── eda1.ipynb                                    # Exploratory analysis
```

---

## Immediate Next Steps

1. **Determine response dynamics**: For each PV/OP tag pair, calculate how long it takes for OP changes to affect PV (response lag analysis)
2. **Analyze historical actions**: For each alarm episode, what actions were taken, in what order, with what magnitudes?
3. **Correlate actions with outcomes**: Which action patterns led to faster alarm resolution?
4. **Feature engineering**: Extract relevant features from PV data before/during alarms
5. **Build simple baseline**: Start with rule-based approach using observed patterns
6. **Iterate**: Progressively improve with more sophisticated methods

---

## Notes for Copilot

- This is exploratory research - be willing to try different approaches
- Prioritize interpretable solutions that can be explained to operators
- Always validate assumptions with data before building complex models
- Document findings and reasoning in notebook markdown cells
- Consider edge cases: What if multiple alarms occur simultaneously?
- The solution must be robust enough for production deployment eventually
