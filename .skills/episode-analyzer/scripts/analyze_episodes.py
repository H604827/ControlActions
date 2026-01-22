#!/usr/bin/env python3
"""
Episode Analyzer - Main Entry Point

Runs comprehensive episode-wise analysis:
1. Rate of change metrics per tag per episode
2. Operating limit deviation checks
3. Operator action analysis

Usage:
    python .skills/episode-analyzer/scripts/analyze_episodes.py \
        --start-date 2025-01-01 --end-date 2025-06-30
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.data_loader import load_all_data, load_trip_data, filter_trip_periods, DataFilterStats


# Default paths
DEFAULT_SSD_PATH = 'DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx'
DEFAULT_TS_PATH = 'DATA/03LIC_1071_JAN_2026.parquet'
DEFAULT_EVENTS_PATH = 'DATA/df_df_events_1071_export.csv'
DEFAULT_LIMITS_PATH = 'DATA/operating_limits.csv'
DEFAULT_TRIP_PATH = 'DATA/Final_List_Trip_Duration.csv'
DEFAULT_OUTPUT_DIR = 'RESULTS/episode-analyzer'


def parse_args():
    parser = argparse.ArgumentParser(description='Episode-wise analysis for alarm episodes')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-trip-filter', action='store_true', help='Disable trip period filtering')
    parser.add_argument('--ssd-file', type=str, default=DEFAULT_SSD_PATH, help='Path to SSD file')
    parser.add_argument('--ts-file', type=str, default=DEFAULT_TS_PATH, help='Path to time series file')
    parser.add_argument('--events-file', type=str, default=DEFAULT_EVENTS_PATH, help='Path to events file')
    parser.add_argument('--operating-limits', type=str, default=DEFAULT_LIMITS_PATH, help='Path to operating limits file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory')
    parser.add_argument('--output-json', action='store_true', help='Also output JSON format')
    return parser.parse_args()


def load_ssd_data(ssd_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load and preprocess SSD data."""
    ssd_df = pd.read_excel(ssd_path)
    
    # Convert datetime columns
    datetime_cols = ['AlarmStart_rounded_minutes', 'AlarmEnd_rounded_minutes', 
                     'Tag_First_Transition_Start_minutes']
    for col in datetime_cols:
        if col in ssd_df.columns:
            ssd_df[col] = pd.to_datetime(ssd_df[col])
    
    # Apply date filters
    if start_date:
        start_dt = pd.to_datetime(start_date)
        ssd_df = ssd_df[ssd_df['AlarmStart_rounded_minutes'] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        ssd_df = ssd_df[ssd_df['AlarmStart_rounded_minutes'] <= end_dt]
    
    return ssd_df


def load_operating_limits(limits_path: str) -> pd.DataFrame:
    """Load operating limits data."""
    limits_df = pd.read_csv(limits_path)
    limits_df = limits_df.set_index('TAG_NAME')
    return limits_df


def get_unique_episodes(ssd_df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique alarm episodes from SSD data."""
    # Group by alarm episode and get earliest transition start
    episodes = ssd_df.groupby(['AlarmStart_rounded_minutes', 'AlarmEnd_rounded_minutes']).agg({
        'Tag_First_Transition_Start_minutes': 'min'
    }).reset_index()
    
    episodes = episodes.rename(columns={
        'Tag_First_Transition_Start_minutes': 'EarliestTransitionStart'
    })
    
    # Add episode ID
    episodes = episodes.sort_values('AlarmStart_rounded_minutes').reset_index(drop=True)
    episodes['EpisodeID'] = range(1, len(episodes) + 1)
    
    # Calculate durations
    episodes['TransitionToAlarmMinutes'] = (
        (episodes['AlarmStart_rounded_minutes'] - episodes['EarliestTransitionStart']).dt.total_seconds() / 60
    )
    episodes['AlarmDurationMinutes'] = (
        (episodes['AlarmEnd_rounded_minutes'] - episodes['AlarmStart_rounded_minutes']).dt.total_seconds() / 60
    )
    episodes['TotalEpisodeDurationMinutes'] = (
        (episodes['AlarmEnd_rounded_minutes'] - episodes['EarliestTransitionStart']).dt.total_seconds() / 60
    )
    
    return episodes


def find_lowest_1071_timestamp(ts_df: pd.DataFrame, alarm_start: pd.Timestamp,
                                alarm_end: pd.Timestamp) -> pd.Timestamp:
    """
    Find the timestamp where 03LIC_1071.PV is at its lowest during alarm period.
    
    Args:
        ts_df: Time series DataFrame with index as timestamp
        alarm_start: Start of alarm period
        alarm_end: End of alarm period
    
    Returns:
        Timestamp of lowest 1071 PV value during alarm period
    """
    target_col = '03LIC_1071.PV'
    if target_col not in ts_df.columns:
        # Fallback to alarm_end if target tag not found
        return alarm_end
    
    mask = (ts_df.index >= alarm_start) & (ts_df.index <= alarm_end)
    alarm_window = ts_df.loc[mask, [target_col]].dropna()
    
    if len(alarm_window) == 0:
        return alarm_end
    
    # Find timestamp of minimum value
    min_idx = alarm_window[target_col].idxmin()
    return min_idx


def compute_percentage_change(ts_df: pd.DataFrame, start_time: pd.Timestamp, 
                               end_time: pd.Timestamp, tag_col: str) -> dict:
    """
    Compute percentage change for a tag from start_time to end_time.
    
    ROC = (final_value - initial_value) / initial_value * 100
    
    Returns metrics dictionary.
    """
    # Get value at start time (or closest after)
    start_mask = (ts_df.index >= start_time)
    start_window = ts_df.loc[start_mask, [tag_col]].dropna()
    
    # Get value at end time (or closest before)
    end_mask = (ts_df.index <= end_time)
    end_window = ts_df.loc[end_mask, [tag_col]].dropna()
    
    if len(start_window) == 0 or len(end_window) == 0:
        return {
            'pct_change': np.nan,
            'start_value': np.nan,
            'end_value': np.nan,
            'absolute_change': np.nan,
            'start_time': str(start_time),
            'end_time': str(end_time),
            'data_available': False
        }
    
    # Get first value after start_time
    start_val = start_window[tag_col].iloc[0]
    start_actual_time = start_window.index[0]
    
    # Get last value before or at end_time
    end_val = end_window[tag_col].iloc[-1]
    end_actual_time = end_window.index[-1]
    
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
        'start_time': str(start_actual_time),
        'end_time': str(end_actual_time),
        'data_available': True
    }


def check_operating_limit_deviation(ts_df: pd.DataFrame, start_time: pd.Timestamp,
                                    end_time: pd.Timestamp, tag_col: str,
                                    limits_df: pd.DataFrame) -> dict:
    """
    Check if a tag deviated from operating limits within a time window.
    
    Returns deviation metrics dictionary.
    """
    # Filter to time window
    mask = (ts_df.index >= start_time) & (ts_df.index <= end_time)
    window_df = ts_df.loc[mask, [tag_col]].dropna()
    
    if len(window_df) == 0:
        return {
            'deviated': False,
            'deviation_type': 'no_data',
            'first_deviation_time': None,
            'deviation_duration_minutes': 0,
            'max_deviation_pct': np.nan,
            'limit_breached': None,
            'data_points': 0
        }
    
    # Check if we have limits for this tag
    if tag_col not in limits_df.index:
        return {
            'deviated': False,
            'deviation_type': 'no_limits_defined',
            'first_deviation_time': None,
            'deviation_duration_minutes': 0,
            'max_deviation_pct': np.nan,
            'limit_breached': None,
            'data_points': len(window_df)
        }
    
    lower_limit = limits_df.loc[tag_col, 'LOWER_LIMIT']
    upper_limit = limits_df.loc[tag_col, 'UPPER_LIMIT']
    
    values = window_df[tag_col].values
    times = window_df.index
    
    # Check deviations
    below_lower = values < lower_limit
    above_upper = values > upper_limit
    deviated = below_lower | above_upper
    
    if not deviated.any():
        return {
            'deviated': False,
            'deviation_type': 'within_limits',
            'first_deviation_time': None,
            'deviation_duration_minutes': 0,
            'max_deviation_pct': 0.0,
            'limit_breached': None,
            'data_points': len(window_df)
        }
    
    # Find first deviation
    first_dev_idx = np.argmax(deviated)
    first_deviation_time = times[first_dev_idx]
    
    # Calculate deviation duration (approximate)
    deviation_count = deviated.sum()
    total_minutes = (times[-1] - times[0]).total_seconds() / 60
    if len(times) > 1:
        deviation_duration = total_minutes * (deviation_count / len(times))
    else:
        deviation_duration = 0
    
    # Calculate max deviation percentage
    limit_range = upper_limit - lower_limit
    if limit_range > 0:
        max_below = np.max(lower_limit - values[below_lower]) if below_lower.any() else 0
        max_above = np.max(values[above_upper] - upper_limit) if above_upper.any() else 0
        max_deviation_pct = max(max_below, max_above) / limit_range * 100
    else:
        max_deviation_pct = np.nan
    
    # Which limit was breached
    if below_lower.any() and above_upper.any():
        limit_breached = 'both'
    elif below_lower.any():
        limit_breached = 'lower'
    else:
        limit_breached = 'upper'
    
    return {
        'deviated': True,
        'deviation_type': 'outside_limits',
        'first_deviation_time': str(first_deviation_time),
        'deviation_duration_minutes': float(deviation_duration),
        'max_deviation_pct': float(max_deviation_pct),
        'limit_breached': limit_breached,
        'data_points': len(window_df)
    }


def analyze_operator_actions(events_df: pd.DataFrame, start_time: pd.Timestamp,
                             end_time: pd.Timestamp, tag_name: str) -> dict:
    """
    Analyze operator actions (CHANGE events) for a tag within a time window.
    
    Returns action metrics dictionary.
    """
    # Filter to CHANGE events in time window
    mask = (
        (events_df['VT_Start'] >= start_time) & 
        (events_df['VT_Start'] <= end_time) &
        (events_df['ConditionName'] == 'CHANGE')
    )
    window_events = events_df[mask].copy()
    
    # Match tag name (handle variations like 03LIC_1071 matching Source or within Description)
    tag_base = tag_name.replace('.PV', '').replace('.OP', '')
    
    # Filter for this tag
    tag_mask = (
        window_events['Source'].str.upper().str.contains(tag_base.upper(), na=False) |
        window_events['Description'].str.upper().str.contains(tag_base.upper(), na=False)
    )
    tag_events = window_events[tag_mask].copy()
    
    if len(tag_events) == 0:
        return {
            'total_changes': 0,
            'op_changes': 0,
            'sp_changes': 0,
            'op_increases': 0,
            'op_decreases': 0,
            'sp_increases': 0,
            'sp_decreases': 0,
            'magnitude_mean': np.nan,
            'magnitude_median': np.nan,
            'magnitude_std': np.nan,
            'magnitude_min': np.nan,
            'magnitude_max': np.nan,
            'cumulative_change': 0.0,
            'first_action_time': None,
            'last_action_time': None
        }
    
    # Classify as OP or SP changes based on Description field
    # The Description field contains: 'OP', 'SP', 'MODE', etc.
    tag_events['is_op'] = tag_events['Description'].str.strip().str.upper() == 'OP'
    tag_events['is_sp'] = tag_events['Description'].str.strip().str.upper() == 'SP'
    
    # Calculate change magnitude and direction
    tag_events['Value'] = pd.to_numeric(tag_events['Value'], errors='coerce')
    tag_events['PrevValue'] = pd.to_numeric(tag_events['PrevValue'], errors='coerce')
    tag_events['change'] = tag_events['Value'] - tag_events['PrevValue']
    tag_events['direction'] = np.where(tag_events['change'] > 0, 'increase', 
                                       np.where(tag_events['change'] < 0, 'decrease', 'no_change'))
    
    # Count by type and direction
    op_events = tag_events[tag_events['is_op']]
    sp_events = tag_events[tag_events['is_sp']]
    
    # Magnitude statistics
    all_magnitudes = tag_events['change'].abs().dropna()
    
    return {
        'total_changes': len(tag_events),
        'op_changes': len(op_events),
        'sp_changes': len(sp_events),
        'op_increases': int((op_events['direction'] == 'increase').sum()),
        'op_decreases': int((op_events['direction'] == 'decrease').sum()),
        'sp_increases': int((sp_events['direction'] == 'increase').sum()),
        'sp_decreases': int((sp_events['direction'] == 'decrease').sum()),
        'magnitude_mean': float(all_magnitudes.mean()) if len(all_magnitudes) > 0 else np.nan,
        'magnitude_median': float(all_magnitudes.median()) if len(all_magnitudes) > 0 else np.nan,
        'magnitude_std': float(all_magnitudes.std()) if len(all_magnitudes) > 0 else np.nan,
        'magnitude_min': float(all_magnitudes.min()) if len(all_magnitudes) > 0 else np.nan,
        'magnitude_max': float(all_magnitudes.max()) if len(all_magnitudes) > 0 else np.nan,
        'cumulative_change': float(tag_events['change'].sum()),
        'first_action_time': str(tag_events['VT_Start'].min()) if len(tag_events) > 0 else None,
        'last_action_time': str(tag_events['VT_Start'].max()) if len(tag_events) > 0 else None
    }


def run_episode_analysis(args):
    """Main analysis function."""
    print(f"📊 Episode Analyzer")
    print(f"=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📁 Loading data...")
    
    # Load SSD data
    ssd_df = load_ssd_data(args.ssd_file, args.start_date, args.end_date)
    print(f"   SSD episodes loaded: {len(ssd_df)} rows")
    
    # Get unique episodes
    episodes_df = get_unique_episodes(ssd_df)
    print(f"   Unique episodes: {len(episodes_df)}")
    
    # Load time series and events using shared loader
    ts_df, events_df, stats = load_all_data(
        ts_path=args.ts_file,
        events_path=args.events_file,
        start_date=args.start_date,
        end_date=args.end_date,
        filter_trips=not args.no_trip_filter,
        verbose=True
    )
    
    # Load operating limits
    limits_df = load_operating_limits(args.operating_limits)
    print(f"   Operating limits loaded: {len(limits_df)} tags")
    
    # Get list of PV tags to analyze
    pv_cols = [col for col in ts_df.columns if col.endswith('.PV')]
    print(f"   PV tags to analyze: {len(pv_cols)}")
    
    # Initialize result containers
    roc_results = []
    limits_results = []
    actions_results = []
    
    print(f"\n🔄 Analyzing {len(episodes_df)} episodes...")
    
    for idx, episode in episodes_df.iterrows():
        episode_id = episode['EpisodeID']
        transition_start = episode['EarliestTransitionStart']
        alarm_start = episode['AlarmStart_rounded_minutes']
        alarm_end = episode['AlarmEnd_rounded_minutes']
        
        if idx % 50 == 0:
            print(f"   Processing episode {episode_id}/{len(episodes_df)}...")
        
        # Find the timestamp where 03LIC_1071.PV is lowest during alarm period
        # This will be the end point for ROC calculation for ALL tags
        lowest_1071_time = find_lowest_1071_timestamp(ts_df, alarm_start, alarm_end)
        
        # Analyze each tag
        for tag_col in pv_cols:
            tag_name = tag_col.replace('.PV', '')
            
            # 1. Percentage change: from transition_start to lowest_1071_time
            # The end time is determined by when 1071 reaches its lowest point
            # but the values used for calculation are from this specific tag
            roc_transition = compute_percentage_change(
                ts_df, transition_start, lowest_1071_time, tag_col
            )
            roc_transition['EpisodeID'] = episode_id
            roc_transition['TagName'] = tag_col
            roc_transition['Period'] = 'transition_to_lowest'
            roc_transition['AlarmStart'] = str(alarm_start)
            roc_transition['AlarmEnd'] = str(alarm_end)
            roc_transition['LowestPointTime'] = str(lowest_1071_time)
            roc_results.append(roc_transition)
            
            # 3. Operating limit deviation (transition_start to alarm_start only)
            limit_dev = check_operating_limit_deviation(
                ts_df, transition_start, alarm_start, tag_col, limits_df
            )
            limit_dev['EpisodeID'] = episode_id
            limit_dev['TagName'] = tag_col
            limit_dev['AlarmStart'] = str(alarm_start)
            limit_dev['AlarmEnd'] = str(alarm_end)
            limits_results.append(limit_dev)
            
            # 4. Operator actions (full episode: transition_start to alarm_end)
            actions = analyze_operator_actions(
                events_df, transition_start, alarm_end, tag_name
            )
            actions['EpisodeID'] = episode_id
            actions['TagName'] = tag_col
            actions['AlarmStart'] = str(alarm_start)
            actions['AlarmEnd'] = str(alarm_end)
            actions_results.append(actions)
    
    # Convert to DataFrames
    roc_df = pd.DataFrame(roc_results)
    limits_dev_df = pd.DataFrame(limits_results)
    actions_df = pd.DataFrame(actions_results)
    
    # Save results
    print(f"\n💾 Saving results...")
    
    # Save episode summary
    episodes_df.to_excel(output_dir / 'episode_summary.xlsx', index=False)
    print(f"   Episode summary: {output_dir / 'episode_summary.xlsx'}")
    
    # Save rate of change
    roc_df.to_excel(output_dir / 'episode_rate_of_change.xlsx', index=False)
    print(f"   Rate of change: {output_dir / 'episode_rate_of_change.xlsx'}")
    
    # Save operating limit deviations
    limits_dev_df.to_excel(output_dir / 'episode_operating_limit_deviations.xlsx', index=False)
    print(f"   Limit deviations: {output_dir / 'episode_operating_limit_deviations.xlsx'}")
    
    # Save operator actions
    actions_df.to_excel(output_dir / 'episode_operator_actions.xlsx', index=False)
    print(f"   Operator actions: {output_dir / 'episode_operator_actions.xlsx'}")
    
    # Create summary grids (episode x tag)
    print(f"\n📊 Creating summary grids...")
    
    # Percentage change grid (transition to lowest 1071 point)
    roc_transition = roc_df[roc_df['Period'] == 'transition_to_lowest']
    roc_grid = roc_transition.pivot(
        index='EpisodeID', columns='TagName', values='pct_change'
    )
    roc_grid.to_excel(output_dir / 'grid_pct_change_transition.xlsx')
    print(f"   Pct change grid: {output_dir / 'grid_pct_change_transition.xlsx'}")
    
    # Operating limit deviation grid
    limits_grid = limits_dev_df.pivot(
        index='EpisodeID', columns='TagName', values='deviated'
    )
    limits_grid.to_excel(output_dir / 'grid_operating_limit_deviation.xlsx')
    print(f"   Limit deviation grid: {output_dir / 'grid_operating_limit_deviation.xlsx'}")
    
    # Combined operator action grid: "OP: X, SP: Y, Total: Z"
    def format_action_counts(row):
        op = int(row.get('op_changes', 0))
        sp = int(row.get('sp_changes', 0))
        total = int(row.get('total_changes', 0))
        if total == 0:
            return ''
        return f"OP: {op}, SP: {sp}, Total: {total}"
    
    actions_df['action_summary'] = actions_df.apply(format_action_counts, axis=1)
    action_grid = actions_df.pivot(
        index='EpisodeID', columns='TagName', values='action_summary'
    )
    action_grid.to_excel(output_dir / 'grid_operator_actions.xlsx')
    print(f"   Action grid: {output_dir / 'grid_operator_actions.xlsx'}")
    
    # Combined direction grid: "OP - Inc: X, Dec: Y | SP - Inc: A, Dec: B"
    def format_direction_counts(row):
        op_inc = int(row.get('op_increases', 0))
        op_dec = int(row.get('op_decreases', 0))
        sp_inc = int(row.get('sp_increases', 0))
        sp_dec = int(row.get('sp_decreases', 0))
        total = op_inc + op_dec + sp_inc + sp_dec
        if total == 0:
            return ''
        return f"OP - Inc: {op_inc}, Dec: {op_dec} | SP - Inc: {sp_inc}, Dec: {sp_dec}"
    
    actions_df['direction_summary'] = actions_df.apply(format_direction_counts, axis=1)
    direction_grid = actions_df.pivot(
        index='EpisodeID', columns='TagName', values='direction_summary'
    )
    direction_grid.to_excel(output_dir / 'grid_action_directions.xlsx')
    print(f"   Direction grid: {output_dir / 'grid_action_directions.xlsx'}")
    
    # Summary statistics
    summary = {
        'analysis_date': str(datetime.now()),
        'date_range': {
            'start': args.start_date,
            'end': args.end_date
        },
        'total_episodes': len(episodes_df),
        'total_tags_analyzed': len(pv_cols),
        'episode_stats': {
            'mean_transition_duration_min': float(episodes_df['TransitionToAlarmMinutes'].mean()),
            'mean_alarm_duration_min': float(episodes_df['AlarmDurationMinutes'].mean()),
            'mean_total_duration_min': float(episodes_df['TotalEpisodeDurationMinutes'].mean())
        },
        'limit_deviations': {
            'episodes_with_any_deviation': int(limits_dev_df.groupby('EpisodeID')['deviated'].any().sum()),
            'tags_with_most_deviations': limits_dev_df[limits_dev_df['deviated']].groupby('TagName').size().nlargest(5).to_dict()
        },
        'operator_actions': {
            'total_actions': int(actions_df['total_changes'].sum()),
            'tags_with_most_actions': actions_df.groupby('TagName')['total_changes'].sum().nlargest(5).to_dict()
        }
    }
    
    if args.output_json:
        with open(output_dir / 'episode_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"   Summary JSON: {output_dir / 'episode_analysis_summary.json'}")
    
    print(f"\n✅ Analysis complete!")
    print(f"\n📈 Summary:")
    print(f"   Episodes analyzed: {summary['total_episodes']}")
    print(f"   Tags analyzed: {summary['total_tags_analyzed']}")
    print(f"   Avg transition duration: {summary['episode_stats']['mean_transition_duration_min']:.1f} min")
    print(f"   Avg alarm duration: {summary['episode_stats']['mean_alarm_duration_min']:.1f} min")
    print(f"   Episodes with limit deviations: {summary['limit_deviations']['episodes_with_any_deviation']}")
    print(f"   Total operator actions: {summary['operator_actions']['total_actions']}")
    
    return summary


if __name__ == '__main__':
    args = parse_args()
    run_episode_analysis(args)
