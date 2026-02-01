#!/usr/bin/env python3
"""
Operator Action Window Analyzer

Analyzes rate of change (ROC) in windows BEFORE each operator action.
For each CHANGE event on target tags (configurable in config.yaml),
calculates ROC for all 1071-related PV tags in three window sizes:
- Long Term (LT): configurable, default 30 minutes before action
- Medium Term (MT): configurable, default 10 minutes before action
- Short Term (ST): configurable, default 3 minutes before action

Usage:
    python .skills/episode-analyzer/scripts/analyze_operator_action_windows.py \
        --start-date 2025-01-01 --end-date 2025-06-30
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yaml

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.data_loader import load_all_data

# Default paths
DEFAULT_TS_PATH = 'DATA/03LIC_1071_JAN_2026.parquet'
DEFAULT_EVENTS_PATH = 'DATA/df_df_events_1071_export.csv'
DEFAULT_OUTPUT_DIR = 'RESULTS/episode-analyzer'
DEFAULT_CONFIG_PATH = '.skills/config.yaml'

# Default values (used if config.yaml is not found)
DEFAULT_TARGET_ACTION_TAGS = ['03LIC_1071', '03LIC_1016', '03PIC_1013']
DEFAULT_WINDOW_SIZES = {
    'LT': 30,  # Long Term: 30 minutes
    'MT': 10,  # Medium Term: 10 minutes
    'ST': 3    # Short Term: 3 minutes
}


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"   ⚠️ Config file not found: {config_path}, using defaults")
        return {}


def get_target_action_tags(config: dict) -> list:
    """Get target action tags from config or use defaults."""
    try:
        return config['operator_action_windows']['target_action_tags']
    except (KeyError, TypeError):
        return DEFAULT_TARGET_ACTION_TAGS


def get_window_sizes(config: dict) -> dict:
    """Get window sizes from config or use defaults."""
    try:
        return config['operator_action_windows']['window_sizes']
    except (KeyError, TypeError):
        return DEFAULT_WINDOW_SIZES


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze ROC in windows before operator actions')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-trip-filter', action='store_true', help='Disable trip period filtering')
    parser.add_argument('--ts-file', type=str, default=DEFAULT_TS_PATH, help='Path to time series file')
    parser.add_argument('--events-file', type=str, default=DEFAULT_EVENTS_PATH, help='Path to events file')
    parser.add_argument('--config', type=str, default=DEFAULT_CONFIG_PATH, help='Path to config.yaml file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--output-json', action='store_true', help='Also output JSON format')
    return parser.parse_args()


def load_operator_actions(events_path: str, target_action_tags: list,
                          start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Load and filter CHANGE events for target tags.
    
    Args:
        events_path: Path to events CSV file
        target_action_tags: List of tags to filter for
        start_date: Filter data after this date (YYYY-MM-DD)
        end_date: Filter data before this date (YYYY-MM-DD)
    
    Returns:
        DataFrame with CHANGE events, VT_Start rounded to nearest second
    """
    events_df = pd.read_csv(events_path, low_memory=False)
    events_df['VT_Start'] = pd.to_datetime(events_df['VT_Start'])
    
    # Round VT_Start to nearest second to match PV/OP data precision
    events_df['VT_Start'] = events_df['VT_Start'].dt.round('s')
    
    # Filter to CHANGE events on target tags
    mask = (
        (events_df['ConditionName'] == 'CHANGE') &
        (events_df['Source'].isin(target_action_tags))
    )
    actions_df = events_df[mask].copy()
    
    # Apply date filters
    if start_date:
        start_dt = pd.to_datetime(start_date)
        actions_df = actions_df[actions_df['VT_Start'] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        actions_df = actions_df[actions_df['VT_Start'] <= end_dt]
    
    # Convert Value and PrevValue to numeric
    actions_df['Value'] = pd.to_numeric(actions_df['Value'], errors='coerce')
    actions_df['PrevValue'] = pd.to_numeric(actions_df['PrevValue'], errors='coerce')
    
    # Calculate magnitude and direction
    actions_df['Magnitude'] = (actions_df['Value'] - actions_df['PrevValue']).abs()
    actions_df['Direction'] = np.where(
        actions_df['Value'] > actions_df['PrevValue'], 'increase',
        np.where(actions_df['Value'] < actions_df['PrevValue'], 'decrease', 'no_change')
    )
    
    # Sort by time
    actions_df = actions_df.sort_values('VT_Start').reset_index(drop=True)
    
    return actions_df


def compute_roc_for_window(ts_df: pd.DataFrame, end_time: pd.Timestamp, 
                           window_minutes: int, tag_col: str) -> dict:
    """
    Compute rate of change (percentage change) for a tag in a window BEFORE end_time.
    
    Args:
        ts_df: Time series DataFrame with index as timestamp
        end_time: End of window (action time)
        window_minutes: Duration of window in minutes
        tag_col: Column name for the tag
    
    Returns:
        Dictionary with ROC metrics
    """
    start_time = end_time - timedelta(minutes=window_minutes)
    
    # Filter to window
    mask = (ts_df.index >= start_time) & (ts_df.index < end_time)
    window_df = ts_df.loc[mask, [tag_col]].dropna()
    
    if len(window_df) < 2:
        return {
            'pct_change': np.nan,
            'start_value': np.nan,
            'end_value': np.nan,
            'absolute_change': np.nan,
            'data_points': len(window_df),
            'data_available': False
        }
    
    # Get first and last values in window
    start_val = window_df[tag_col].iloc[0]
    end_val = window_df[tag_col].iloc[-1]
    
    # Calculate absolute change
    absolute_change = end_val - start_val
    
    # Calculate percentage change relative to initial value
    if start_val != 0:
        pct_change = (absolute_change / abs(start_val)) * 100
    else:
        pct_change = np.nan if absolute_change == 0 else np.inf * np.sign(absolute_change)
    
    return {
        'pct_change': float(pct_change) if not np.isinf(pct_change) else None,
        'start_value': float(start_val),
        'end_value': float(end_val),
        'absolute_change': float(absolute_change),
        'data_points': len(window_df),
        'data_available': True
    }


def compute_roc_for_all_tags(ts_df: pd.DataFrame, end_time: pd.Timestamp,
                             window_sizes: dict, pv_cols: list) -> dict:
    """
    Compute ROC for all PV tags for all window sizes at a given timestamp.
    
    This function is used for caching - we compute all ROC values once per unique timestamp.
    
    Args:
        ts_df: Time series DataFrame with index as timestamp
        end_time: End of window (action time)
        window_sizes: Dict mapping window names to durations in minutes
        pv_cols: List of PV column names
    
    Returns:
        Dictionary mapping (window_name, tag_col) to ROC result dict
    """
    results = {}
    for window_name, window_minutes in window_sizes.items():
        for tag_col in pv_cols:
            roc = compute_roc_for_window(ts_df, end_time, window_minutes, tag_col)
            results[(window_name, tag_col)] = roc
    return results


def run_analysis(args):
    """Main analysis function."""
    # Load configuration
    config = load_config(args.config)
    target_action_tags = get_target_action_tags(config)
    window_sizes = get_window_sizes(config)
    
    print(f"📊 Operator Action Window Analyzer")
    print(f"=" * 60)
    print(f"   Config file: {args.config}")
    print(f"   Target action tags: {', '.join(target_action_tags)}")
    window_str = ', '.join([f"{k}={v}min" for k, v in window_sizes.items()])
    print(f"   Window sizes: {window_str}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load operator actions
    print(f"\n📁 Loading operator actions...")
    actions_df = load_operator_actions(args.events_file, target_action_tags, 
                                        args.start_date, args.end_date)
    print(f"   Total CHANGE events loaded: {len(actions_df)}")
    print(f"   By tag:")
    for tag in target_action_tags:
        count = len(actions_df[actions_df['Source'] == tag])
        print(f"      {tag}: {count}")
    
    # Load time series data
    print(f"\n📁 Loading time series data...")
    ts_df, _, stats = load_all_data(
        ts_path=args.ts_file,
        events_path=None,
        start_date=args.start_date,
        end_date=args.end_date,
        filter_trips=not args.no_trip_filter,
        verbose=True
    )
    
    # Get list of PV tags to analyze
    pv_cols = [col for col in ts_df.columns if col.endswith('.PV')]
    print(f"   PV tags to analyze: {len(pv_cols)}")
    
    # Get unique timestamps and count duplicates
    unique_timestamps = actions_df['VT_Start'].unique()
    total_actions = len(actions_df)
    unique_count = len(unique_timestamps)
    duplicate_count = total_actions - unique_count
    print(f"\n   Unique timestamps: {unique_count}")
    print(f"   Duplicate timestamps: {duplicate_count} (will reuse cached ROC)")
    
    # Pre-compute ROC for all unique timestamps (caching optimization)
    print(f"\n🔄 Pre-computing ROC for {unique_count} unique timestamps...")
    roc_cache = {}
    for i, ts in enumerate(unique_timestamps):
        if i % 200 == 0:
            print(f"   Caching timestamp {i+1}/{unique_count}...")
        roc_cache[ts] = compute_roc_for_all_tags(ts_df, ts, window_sizes, pv_cols)
    
    # Build results using cached values
    print(f"\n🔄 Building results for {total_actions} operator actions...")
    results = []
    
    for idx, action in actions_df.iterrows():
        action_time = action['VT_Start']
        action_tag = action['Source']
        magnitude = action['Magnitude']
        direction = action['Direction']
        description = action['Description']  # OP, SP, MODE, etc.
        
        if idx % 1000 == 0:
            print(f"   Processing action {idx+1}/{total_actions}...")
        
        # Get cached ROC values for this timestamp
        cached_roc = roc_cache[action_time]
        
        # Build rows for each PV tag
        for pv_col in pv_cols:
            row = {
                'ActionID': idx + 1,
                'ActionTime': str(action_time),
                'ActionTag': action_tag,
                'ActionType': description,
                'Magnitude': magnitude,
                'Direction': direction,
                'PVTag': pv_col
            }
            
            # Get ROC from cache for each window size
            for window_name in window_sizes.keys():
                roc = cached_roc[(window_name, pv_col)]
                row[f'{window_name}_ROC'] = roc['pct_change']
                row[f'{window_name}_start_value'] = roc['start_value']
                row[f'{window_name}_end_value'] = roc['end_value']
                row[f'{window_name}_abs_change'] = roc['absolute_change']
                row[f'{window_name}_data_points'] = roc['data_points']
            
            results.append(row)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Save full results
    print(f"\n💾 Saving results...")
    output_file = output_dir / 'operator_action_window_roc.xlsx'
    results_df.to_excel(output_file, index=False)
    print(f"   Full results: {output_file}")
    
    # Create summary view with just ROC values
    # Dynamically build column list based on window_sizes
    roc_cols = [f'{wn}_ROC' for wn in window_sizes.keys()]
    summary_cols = ['ActionID', 'ActionTime', 'ActionTag', 'ActionType', 
                    'Magnitude', 'Direction', 'PVTag'] + roc_cols
    summary_df = results_df[summary_cols].copy()
    summary_file = output_dir / 'operator_action_window_roc_summary.xlsx'
    summary_df.to_excel(summary_file, index=False)
    print(f"   Summary: {summary_file}")
    
    # Create pivot tables for quick analysis
    print(f"\n📊 Creating pivot tables...")
    
    for window_name in window_sizes.keys():
        roc_col = f'{window_name}_ROC'
        pivot = results_df.pivot(index='ActionID', columns='PVTag', values=roc_col)
        pivot_file = output_dir / f'grid_action_pv_{window_name.lower()}_roc.xlsx'
        pivot.to_excel(pivot_file)
        print(f"   {window_name} ROC grid: {pivot_file}")
    
    # Summary statistics
    summary_stats = {
        'analysis_date': str(datetime.now()),
        'date_range': {
            'start': args.start_date,
            'end': args.end_date
        },
        'window_sizes_minutes': window_sizes,
        'total_actions_analyzed': len(actions_df),
        'unique_timestamps': unique_count,
        'duplicate_timestamps': duplicate_count,
        'actions_by_tag': actions_df['Source'].value_counts().to_dict(),
        'actions_by_type': actions_df['Description'].value_counts().to_dict(),
        'actions_by_direction': actions_df['Direction'].value_counts().to_dict(),
        'total_pv_tags': len(pv_cols),
        'magnitude_stats': {
            'mean': float(actions_df['Magnitude'].mean()),
            'median': float(actions_df['Magnitude'].median()),
            'std': float(actions_df['Magnitude'].std()),
            'min': float(actions_df['Magnitude'].min()),
            'max': float(actions_df['Magnitude'].max())
        },
        'roc_stats': {}
    }
    
    # Build ROC stats dynamically
    for window_name in window_sizes.keys():
        roc_col = f'{window_name}_ROC'
        data_col = f'{window_name}_data_points'
        summary_stats['roc_stats'][window_name] = {
            'mean': float(results_df[roc_col].mean()),
            'std': float(results_df[roc_col].std()),
            'data_coverage_pct': float((results_df[data_col] > 1).mean() * 100)
        }
    
    if args.output_json:
        with open(output_dir / 'operator_action_window_summary.json', 'w') as f:
            json.dump(summary_stats, f, indent=2, default=str)
        print(f"   Summary JSON: {output_dir / 'operator_action_window_summary.json'}")
    
    print(f"\n✅ Analysis complete!")
    print(f"\n📈 Summary:")
    print(f"   Total actions analyzed: {summary_stats['total_actions_analyzed']}")
    print(f"   Unique timestamps: {summary_stats['unique_timestamps']}")
    print(f"   Duplicate timestamps (reused cache): {summary_stats['duplicate_timestamps']}")
    print(f"   PV tags analyzed: {summary_stats['total_pv_tags']}")
    print(f"   Actions by tag:")
    for tag, count in summary_stats['actions_by_tag'].items():
        print(f"      {tag}: {count}")
    print(f"   Actions by direction:")
    for direction, count in summary_stats['actions_by_direction'].items():
        print(f"      {direction}: {count}")
    print(f"   Avg magnitude: {summary_stats['magnitude_stats']['mean']:.2f}")
    
    return summary_stats


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
