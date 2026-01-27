#!/usr/bin/env python3
"""
Operator Action Analyzer for Alarm Episodes.

Analyzes CHANGE events (operator actions) for each tag within each episode:
- Number of SP and OP changes
- Direction breakdown (increases vs decreases)
- Magnitude statistics (mean, median, std, min, max)
- Cumulative change

Usage:
    python .skills/episode-analyzer/scripts/analyze_operator_actions.py \
        --start-date 2025-01-01 --end-date 2025-06-30
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.data_loader import load_all_data
from shared.episode_utils import (
    load_ssd_data,
    load_ground_truth_with_fallback,
    get_unique_episodes,
    get_unique_tags_from_ssd
)


DEFAULT_EVENTS_PATH = 'DATA/df_df_events_1071_export.csv'
DEFAULT_OUTPUT_DIR = 'RESULTS'


def parse_args():
    parser = argparse.ArgumentParser(description='Analyze operator actions per episode')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-trip-filter', action='store_true', help='Disable trip filtering')
    parser.add_argument('--ssd-file', type=str, default='DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx')
    parser.add_argument('--events-file', type=str, default=DEFAULT_EVENTS_PATH)
    parser.add_argument('--ground-truth', type=str, default='DATA/Updated Ground truth -Adnoc RCA - recent(all_episode_top5_test_validated).csv',
                        help='Path to ground truth CSV with AlarmStart_rounded column for episode filtering')
    parser.add_argument('--no-ground-truth-filter', action='store_true',
                        help='Disable filtering episodes to those in ground truth file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--output-json', action='store_true')
    return parser.parse_args()


def analyze_actions_for_tag(events_df: pd.DataFrame, start_time: pd.Timestamp,
                            end_time: pd.Timestamp, tag_base: str) -> dict:
    """
    Analyze operator actions for a specific tag within an episode.
    
    Args:
        events_df: Events dataframe with CHANGE events
        start_time: Episode start (transition start)
        end_time: Episode end (alarm end)
        tag_base: Base tag name (without .PV/.OP suffix)
    
    Returns:
        Dictionary with action metrics
    """
    # Filter to CHANGE events in time window
    mask = (
        (events_df['VT_Start'] >= start_time) & 
        (events_df['VT_Start'] <= end_time) &
        (events_df['ConditionName'] == 'CHANGE')
    )
    window_events = events_df[mask].copy()
    
    # Match tag (case-insensitive)
    tag_upper = tag_base.upper()
    tag_mask = (
        window_events['Source'].str.upper().str.contains(tag_upper, na=False, regex=False)
    )
    tag_events = window_events[tag_mask].copy()
    
    base_result = {
        'total_changes': 0,
        'op_changes': 0,
        'sp_changes': 0,
        'mode_changes': 0,
        'other_changes': 0,
        'op_increases': 0,
        'op_decreases': 0,
        'sp_increases': 0,
        'sp_decreases': 0,
        'magnitude_mean': np.nan,
        'magnitude_median': np.nan,
        'magnitude_std': np.nan,
        'magnitude_min': np.nan,
        'magnitude_max': np.nan,
        'op_magnitude_mean': np.nan,
        'sp_magnitude_mean': np.nan,
        'cumulative_change': 0.0,
        'op_cumulative_change': 0.0,
        'sp_cumulative_change': 0.0,
        'first_action_time': None,
        'last_action_time': None,
        'action_sources': []
    }
    
    if len(tag_events) == 0:
        return base_result
    
    # Classify event type based on Source field
    # OP changes: Source contains .OP or ends with _OP
    # SP changes: Source contains .SP or ends with _SP, or Description mentions SP/SETPOINT
    
    def classify_action(row):
        desc = str(row.get('Description', '')).strip().upper()
        
        # Description field contains the type directly: 'OP', 'SP', 'MODE', etc.
        if desc == 'OP':
            return 'OP'
        elif desc == 'SP':
            return 'SP'
        elif desc == 'MODE':
            return 'MODE'
        else:
            return 'OTHER'
    
    tag_events['action_type'] = tag_events.apply(classify_action, axis=1)
    
    # Calculate change magnitude and direction
    tag_events['Value'] = pd.to_numeric(tag_events['Value'], errors='coerce')
    tag_events['PrevValue'] = pd.to_numeric(tag_events['PrevValue'], errors='coerce')
    tag_events['change'] = tag_events['Value'] - tag_events['PrevValue']
    tag_events['abs_change'] = tag_events['change'].abs()
    tag_events['direction'] = np.where(
        tag_events['change'] > 0, 'increase',
        np.where(tag_events['change'] < 0, 'decrease', 'no_change')
    )
    
    # Split by action type
    op_events = tag_events[tag_events['action_type'] == 'OP']
    sp_events = tag_events[tag_events['action_type'] == 'SP']
    mode_events = tag_events[tag_events['action_type'] == 'MODE']
    other_events = tag_events[tag_events['action_type'] == 'OTHER']
    
    # Overall magnitude stats (only for numeric changes, exclude MODE)
    numeric_events = tag_events[tag_events['action_type'].isin(['OP', 'SP'])]
    all_magnitudes = numeric_events['abs_change'].dropna()
    
    result = {
        'total_changes': len(tag_events),
        'op_changes': len(op_events),
        'sp_changes': len(sp_events),
        'mode_changes': len(mode_events),
        'other_changes': len(other_events),
        'op_increases': int((op_events['direction'] == 'increase').sum()),
        'op_decreases': int((op_events['direction'] == 'decrease').sum()),
        'sp_increases': int((sp_events['direction'] == 'increase').sum()),
        'sp_decreases': int((sp_events['direction'] == 'decrease').sum()),
        'magnitude_mean': float(all_magnitudes.mean()) if len(all_magnitudes) > 0 else np.nan,
        'magnitude_median': float(all_magnitudes.median()) if len(all_magnitudes) > 0 else np.nan,
        'magnitude_std': float(all_magnitudes.std()) if len(all_magnitudes) > 0 else np.nan,
        'magnitude_min': float(all_magnitudes.min()) if len(all_magnitudes) > 0 else np.nan,
        'magnitude_max': float(all_magnitudes.max()) if len(all_magnitudes) > 0 else np.nan,
        'op_magnitude_mean': float(op_events['abs_change'].mean()) if len(op_events) > 0 else np.nan,
        'sp_magnitude_mean': float(sp_events['abs_change'].mean()) if len(sp_events) > 0 else np.nan,
        'cumulative_change': float(tag_events['change'].sum()),
        'op_cumulative_change': float(op_events['change'].sum()) if len(op_events) > 0 else 0.0,
        'sp_cumulative_change': float(sp_events['change'].sum()) if len(sp_events) > 0 else 0.0,
        'first_action_time': str(tag_events['VT_Start'].min()),
        'last_action_time': str(tag_events['VT_Start'].max()),
        'action_sources': list(tag_events['Source'].unique())
    }
    
    return result


def run_analysis(args):
    print(f"📊 Operator Action Analysis per Episode")
    print(f"=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📁 Loading data...")
    ssd_df = load_ssd_data(args.ssd_file, args.start_date, args.end_date)
    
    # Load ground truth alarm starts for filtering (unless disabled)
    ground_truth_alarm_starts = None
    if not args.no_ground_truth_filter:
        ground_truth_alarm_starts = load_ground_truth_with_fallback(args.ground_truth, verbose=True)
    else:
        print(f"   Ground truth filtering disabled")
    
    episodes_df = get_unique_episodes(ssd_df, ground_truth_alarm_starts)
    print(f"   Episodes: {len(episodes_df)}")
    
    # Get unique tags from SSD to analyze
    tag_bases = get_unique_tags_from_ssd(ssd_df)
    print(f"   Tags to analyze: {len(tag_bases)}")
    
    _, events_df, stats = load_all_data(
        ts_path=None,
        events_path=args.events_file,
        start_date=args.start_date,
        end_date=args.end_date,
        filter_trips=not args.no_trip_filter,
        verbose=True
    )
    
    # Filter to CHANGE events only for efficiency
    change_events = events_df[events_df['ConditionName'] == 'CHANGE'].copy()
    print(f"   CHANGE events in period: {len(change_events)}")
    
    # Analyze actions
    print(f"\n🔄 Analyzing operator actions...")
    results = []
    
    for idx, episode in episodes_df.iterrows():
        ep_id = episode['EpisodeID']
        transition_start = episode['EarliestTransitionStart']
        alarm_end = episode['AlarmEnd_rounded_minutes']
        alarm_start = episode['AlarmStart_rounded_minutes']
        
        if idx % 50 == 0:
            print(f"   Episode {ep_id}/{len(episodes_df)}...")
        
        for tag_base in tag_bases:
            actions = analyze_actions_for_tag(change_events, transition_start, alarm_end, tag_base)
            actions.update({
                'EpisodeID': ep_id,
                'TagName': tag_base,
                'TransitionStart': str(transition_start),
                'AlarmStart': str(alarm_start),
                'AlarmEnd': str(alarm_end)
            })
            # Convert action_sources list to string for Excel
            actions['action_sources'] = ', '.join(actions['action_sources'])
            results.append(actions)
    
    # Convert to DataFrame
    actions_df = pd.DataFrame(results)
    
    # Save detailed results
    output_file = output_dir / 'episode_operator_actions_detailed.xlsx'
    actions_df.to_excel(output_file, index=False)
    print(f"\n💾 Saved: {output_file}")
    
    # Create summary grids
    
    # Combined action grid: "OP: X, SP: Y, Total: Z"
    def format_action_counts(row):
        op = int(row.get('op_changes', 0))
        sp = int(row.get('sp_changes', 0))
        total = int(row.get('total_changes', 0))
        if total == 0:
            return ''
        return f"OP: {op}, SP: {sp}, Total: {total}"
    
    actions_df['action_summary'] = actions_df.apply(format_action_counts, axis=1)
    action_grid = actions_df.pivot(index='EpisodeID', columns='TagName', values='action_summary')
    action_grid.to_excel(output_dir / 'grid_operator_actions.xlsx')
    
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
    direction_grid = actions_df.pivot(index='EpisodeID', columns='TagName', values='direction_summary')
    direction_grid.to_excel(output_dir / 'grid_action_directions.xlsx')
    
    print(f"   Saved action and direction grids")
    
    # Summary statistics
    tags_with_actions = actions_df[actions_df['total_changes'] > 0].groupby('TagName').size()
    episodes_with_actions = actions_df.groupby('EpisodeID')['total_changes'].sum()
    
    summary = {
        'total_episodes': len(episodes_df),
        'total_tags_analyzed': len(tag_bases),
        'episodes_with_actions': int((episodes_with_actions > 0).sum()),
        'episodes_without_actions': int((episodes_with_actions == 0).sum()),
        'total_actions': int(actions_df['total_changes'].sum()),
        'total_op_changes': int(actions_df['op_changes'].sum()),
        'total_sp_changes': int(actions_df['sp_changes'].sum()),
        'op_direction_summary': {
            'total_increases': int(actions_df['op_increases'].sum()),
            'total_decreases': int(actions_df['op_decreases'].sum())
        },
        'sp_direction_summary': {
            'total_increases': int(actions_df['sp_increases'].sum()),
            'total_decreases': int(actions_df['sp_decreases'].sum())
        },
        'most_operated_tags': tags_with_actions.nlargest(10).to_dict(),
        'magnitude_stats': {
            'overall_mean': float(actions_df['magnitude_mean'].mean()),
            'overall_median': float(actions_df['magnitude_median'].median()),
            'max_single_change': float(actions_df['magnitude_max'].max())
        }
    }
    
    if args.output_json:
        with open(output_dir / 'operator_action_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete!")
    print(f"\n📈 Summary:")
    print(f"   Episodes with actions: {summary['episodes_with_actions']}/{summary['total_episodes']}")
    print(f"   Total actions: {summary['total_actions']}")
    print(f"   OP changes: {summary['total_op_changes']} (↑{summary['op_direction_summary']['total_increases']} / ↓{summary['op_direction_summary']['total_decreases']})")
    print(f"   SP changes: {summary['total_sp_changes']} (↑{summary['sp_direction_summary']['total_increases']} / ↓{summary['sp_direction_summary']['total_decreases']})")
    
    if summary['most_operated_tags']:
        print(f"\n   Most operated tags:")
        for tag, count in list(summary['most_operated_tags'].items())[:5]:
            print(f"     {tag}: {count} episodes with actions")
    
    return summary


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
