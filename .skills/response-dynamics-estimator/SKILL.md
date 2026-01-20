---
name: response-dynamics-estimator
description: Estimate process response dynamics between OP and PV tags to determine how long control actions take to affect process variables. Use when calculating response lag, identifying cause-effect timing, or determining how quickly OP changes affect PV values in the Control Actions project.
metadata:
  author: control-actions-team
  version: "2.0"
  target-alarm: "03LIC_1071"
  updated: "2026-01-20"
compatibility: Requires scipy, numpy, pandas. Designed for minute-wise time series data.
---

# Response Dynamics Estimator

## What's New in v2.0 (Jan 2026)

- **Trip period filtering**: Automatically excludes plant shutdown periods for cleaner analysis
- **Date range filtering**: Analyze recent data only (recommended for better accuracy)
- **Time-segmented comparison**: Compare dynamics across years to detect process changes
- **Improved reporting**: Shows filtering stats and data range in outputs

## When to Use This Skill

Use this skill when you need to:
- Determine how long it takes for an OP change to affect the corresponding PV
- Calculate the response lag/delay for each PV/OP tag pair
- Understand process dynamics for timing control actions
- Avoid overcorrection by knowing when to expect results

## Why This Matters

For autonomous control actions, we need to know:
1. **When to expect results**: After changing OP, how long until PV responds?
2. **When to take next action**: Wait for response before acting again
3. **Avoid overcorrection**: Don't keep changing OP if PV hasn't had time to respond

## Methods Overview

### 1. Cross-Correlation Analysis
Find the time lag where OP and PV are most correlated.
- **Best for**: Continuous, noisy data
- **Output**: Optimal lag in minutes

### 2. Step Response Identification
Analyze how PV responds after discrete OP changes (from CHANGE events).
- **Best for**: Discrete operator actions
- **Output**: Response time, settling time

### 3. Transfer Function Estimation
Model the OP→PV relationship as a first-order system.
- **Best for**: Understanding process behavior
- **Output**: Time constant, gain, dead time

## Step-by-Step Instructions

### Step 1: Load Data

```python
import pandas as pd
import numpy as np
from scipy import signal

# Load time series
op_pv_df = pd.read_parquet('DATA/03LIC_1071_JAN_2026.parquet')
op_pv_df.set_index('TimeStamp', inplace=True)
op_pv_df.sort_index(inplace=True)

# Target tag
target_tag = '03LIC_1071'
pv_col = f'{target_tag}.PV'
op_col = f'{target_tag}.OP'
```

### Step 2: Cross-Correlation Analysis

```python
def estimate_lag_crosscorr(pv_series, op_series, max_lag_minutes=60):
    """
    Estimate lag using cross-correlation.
    Returns optimal lag in minutes where PV best correlates with past OP.
    """
    # Clean data
    df = pd.DataFrame({'pv': pv_series, 'op': op_series}).dropna()
    pv = df['pv'].values
    op = df['op'].values
    
    # Normalize
    pv = (pv - pv.mean()) / pv.std()
    op = (op - op.mean()) / op.std()
    
    # Compute cross-correlation
    correlation = signal.correlate(pv, op, mode='full')
    lags = signal.correlation_lags(len(pv), len(op), mode='full')
    
    # Focus on positive lags (OP leads PV)
    positive_mask = (lags >= 0) & (lags <= max_lag_minutes)
    lags_pos = lags[positive_mask]
    corr_pos = correlation[positive_mask]
    
    # Find peak
    peak_idx = np.argmax(np.abs(corr_pos))
    optimal_lag = lags_pos[peak_idx]
    peak_corr = corr_pos[peak_idx] / len(pv)
    
    return {
        'optimal_lag_minutes': int(optimal_lag),
        'correlation_strength': float(peak_corr),
        'lags': lags_pos.tolist(),
        'correlations': (corr_pos / len(pv)).tolist()
    }
```

### Step 3: Step Response Analysis

Run the step response script for detailed analysis:
```bash
python .skills/response-dynamics-estimator/scripts/analyze_step_response.py --tag 03LIC_1071
```

Or manually:
```python
def analyze_op_step_response(op_pv_df, events_df, tag_name, 
                             window_before=10, window_after=30):
    """
    Analyze PV response after discrete OP changes (CHANGE events).
    """
    pv_col = f'{tag_name}.PV'
    
    # Get CHANGE events for this tag
    changes = events_df[
        (events_df['Source'] == tag_name) & 
        (events_df['ConditionName'] == 'CHANGE')
    ].copy()
    
    responses = []
    
    for _, event in changes.iterrows():
        event_time = event['VT_Start']
        
        # Extract window around event
        start = event_time - pd.Timedelta(minutes=window_before)
        end = event_time + pd.Timedelta(minutes=window_after)
        
        window_data = op_pv_df.loc[start:end, pv_col]
        
        if len(window_data) < 5:
            continue
            
        # Calculate response characteristics
        pv_before = window_data[:window_before].mean()
        pv_after = window_data[window_before:].values
        
        # Find when PV starts responding (crosses 10% of final change)
        # Find when PV settles (within 5% of final value)
        # ... detailed analysis
        
        responses.append({
            'event_time': event_time,
            'pv_before': pv_before,
            'pv_trajectory': pv_after.tolist()
        })
    
    return responses
```

### Step 4: Estimate First-Order Time Constant

```python
def estimate_first_order_params(pv_series, op_series):
    """
    Fit a first-order model: PV responds to OP with time constant tau.
    Model: dPV/dt = (K * OP - PV) / tau
    """
    from scipy.optimize import curve_fit
    
    # This is a simplified approach
    # For proper system identification, use scipy.signal or control library
    
    # Calculate rate of change
    dt = 1  # 1 minute
    dpv_dt = np.diff(pv_series) / dt
    
    # Simple linear regression to estimate tau and K
    # dpv/dt ≈ (K*op - pv) / tau
    # Rearranged: dpv/dt * tau + pv ≈ K * op
    
    # ... detailed implementation in scripts
    
    return {
        'time_constant_minutes': None,  # tau
        'gain': None,  # K
        'dead_time_minutes': None  # delay before response starts
    }
```

### Step 5: Generate Response Dynamics Report

Run the comprehensive analysis:
```bash
python .skills/response-dynamics-estimator/scripts/estimate_dynamics.py
```

## Expected Results for Gas Processes

Based on your hypothesis that gas processes respond quickly:
- **Expected lag**: 1-10 minutes for most tags
- **Typical time constant**: 2-5 minutes for fast loops
- **Dead time**: Usually < 2 minutes for pressure/flow, may be longer for temperature

## Command Line Usage

### Basic Usage (with trip filtering, full data)
```bash
cd /home/h604827/ControlActions
conda activate adnoc
python .skills/response-dynamics-estimator/scripts/estimate_dynamics.py
```

### Analyze Recent Data Only (RECOMMENDED)
```bash
# Last 6 months - most relevant for current process behavior
python .skills/response-dynamics-estimator/scripts/estimate_dynamics.py --recent

# Last 12 months
python .skills/response-dynamics-estimator/scripts/estimate_dynamics.py --last-year

# Custom date range
python .skills/response-dynamics-estimator/scripts/estimate_dynamics.py \
    --start-date 2024-01-01 --end-date 2025-06-30
```

### Compare Dynamics Across Years
```bash
# Detect if process dynamics have changed over time
python .skills/response-dynamics-estimator/scripts/estimate_dynamics.py --compare-years
```

### Disable Trip Filtering (not recommended)
```bash
python .skills/response-dynamics-estimator/scripts/estimate_dynamics.py --no-trip-filter
```

## Key Outputs

After running this skill:
1. **Lag estimates** for each PV/OP pair
2. **Confidence scores** for each estimate
3. **Visualization** of cross-correlation results
4. **Step response plots** around operator actions
5. **Filtering statistics** showing data quality

## Interpretation Guidelines

| Lag (minutes) | Interpretation | Action Timing |
|---------------|----------------|---------------|
| 0-2 | Very fast response | Can take rapid sequential actions |
| 2-5 | Fast response | Wait 5 min before next action |
| 5-15 | Moderate response | Wait 15 min, observe trend |
| >15 | Slow response | Plan actions carefully, avoid overcorrection |

## Important Considerations

### Why Filter Trip Periods?
During plant trips/shutdowns:
- PV values may flatline or show abnormal patterns
- OP changes are for startup/shutdown, not normal control
- Including trips will skew correlation estimates

### Why Use Recent Data?
Over 3+ years, many things change:
- Controller tuning parameters
- Equipment modifications
- Process conditions
- Sensor calibration drift

**Recommendation**: Use `--recent` or `--last-year` for most accurate current dynamics.

### Limitations of Linear Correlation
Cross-correlation assumes linear relationships. Industrial processes may have:
- Nonlinear valve characteristics
- Operating point-dependent gain
- Multi-variable interactions

For complex cases, consider the step response analysis which is more robust.
- **Negative lag**: Indicates PV might be leading OP (reverse causation or feedback)
