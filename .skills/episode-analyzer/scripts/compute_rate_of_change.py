#!/usr/bin/env python3
"""
Percentage Change Calculator for Alarm Episodes.

Computes percentage change for each tag within each episode:
- Window: from TransitionStart to the point where 03LIC_1071.PV is lowest during alarm period
- Calculation: (final_value - initial_value) / initial_value * 100

Usage:
    python .skills/episode-analyzer/scripts/compute_rate_of_change.py \\
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


DEFAULT_SSD_PATH = 'DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx'
DEFAULT_TS_PATH = 'DATA/03LIC_1071_JAN_2026.parquet'
DEFAULT_OUTPUT_DIR = 'RESULTS'


def parse_args():
    parser = argparse.ArgumentParser(description='Compute percentage change metrics per episode')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-trip-filter', action='store_true', help='Disable trip filtering')
    parser.add_argument('--ssd-file', type=str, default=DEFAULT_SSD_PATH)
    parser.add_argument('--ts-file', type=str, default=DEFAULT_TS_PATH)
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--output-json', action='store_true')
    return parser.parse_args()


def load_ssd_data(path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    ssd_df = pd.read_excel(path)
    for col in ['AlarmStart_rounded_minutes', 'AlarmEnd_rounded_minutes', 'Tag_First_Transition_Start_minutes']:
        if col in ssd_df.columns:
            ssd_df[col] = pd.to_datetime(ssd_df[col])
    
    if start_date:
        ssd_df = ssd_df[ssd_df['AlarmStart_rounded_minutes'] >= pd.to_datetime(start_date)]
    if end_date:
        ssd_df = ssd_df[ssd_df['AlarmStart_rounded_minutes'] <= pd.to_datetime(end_date)]
    
    return ssd_df


def get_unique_episodes(ssd_df: pd.DataFrame) -> pd.DataFrame:
    episodes = ssd_df.groupby(['AlarmStart_rounded_minutes', 'AlarmEnd_rounded_minutes']).agg({
        'Tag_First_Transition_Start_minutes': 'min'
    }).reset_index()
    episodes = episodes.rename(columns={'Tag_First_Transition_Start_minutes': 'EarliestTransitionStart'})
    episodes = episodes.sort_values('AlarmStart_rounded_minutes').reset_index(drop=True)
    episodes['EpisodeID'] = range(1, len(episodes) + 1)
    return episodes


def find_lowest_1071_timestamp(ts_df: pd.DataFrame, alarm_start: pd.Timestamp,
                                alarm_end: pd.Timestamp) -> pd.Timestamp:
    """
    Find the timestamp where 03LIC_1071.PV is at its lowest during alarm period.
    """
    target_col = '03LIC_1071.PV'
    if target_col not in ts_df.columns:
        return alarm_end
    
    mask = (ts_df.index >= alarm_start) & (ts_df.index <= alarm_end)
    alarm_window = ts_df.loc[mask, [target_col]].dropna()
    
    if len(alarm_window) == 0:
        return alarm_end
    
    return alarm_window[target_col].idxmin()


def compute_percentage_change(ts_df: pd.DataFrame, start_time: pd.Timestamp, 
                               end_time: pd.Timestamp, tag_col: str) -> dict:
    """
    Compute percentage change for a tag from start_time to end_time.
    
    ROC = (final_value - initial_value) / initial_value * 100
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
    
    start_val = start_window[tag_col].iloc[0]
    start_actual_time = start_window.index[0]
    end_val = end_window[tag_col].iloc[-1]
    end_actual_time = end_window.index[-1]
    
    absolute_change = end_val - start_val
    
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


def run_analysis(args):
    print(f"📊 Percentage Change Analysis")
    print(f"=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\\n📁 Loading data...")
    ssd_df = load_ssd_data(args.ssd_file, args.start_date, args.end_date)
    episodes_df = get_unique_episodes(ssd_df)
    print(f"   Episodes: {len(episodes_df)}")
    
    ts_df, _, stats = load_all_data(
        ts_path=args.ts_file,
        events_path=None,
        start_date=args.start_date,
        end_date=args.end_date,
        filter_trips=not args.no_trip_filter,
        verbose=True
    )
    
    pv_cols = [col for col in ts_df.columns if col.endswith('.PV')]
    print(f"   PV tags: {len(pv_cols)}")
    
    # Compute metrics
    print(f"\\n🔄 Computing percentage change metrics...")
    results = []
    
    for idx, episode in episodes_df.iterrows():
        ep_id = episode['EpisodeID']
        transition_start = episode['EarliestTransitionStart']
        alarm_start = episode['AlarmStart_rounded_minutes']
        alarm_end = episode['AlarmEnd_rounded_minutes']
        
        if idx % 50 == 0:
            print(f"   Episode {ep_id}/{len(episodes_df)}...")
        
        # Find lowest 1071 point - this determines end time for ALL tags
        lowest_time = find_lowest_1071_timestamp(ts_df, alarm_start, alarm_end)
        
        for tag_col in pv_cols:
            # Calculate percentage change from transition_start to lowest_time
            pct_result = compute_percentage_change(ts_df, transition_start, lowest_time, tag_col)
            pct_result.update({
                'EpisodeID': ep_id,
                'TagName': tag_col,
                'TransitionStart': str(transition_start),
                'AlarmStart': str(alarm_start),
                'AlarmEnd': str(alarm_end),
                'LowestPointTime': str(lowest_time)
            })
            results.append(pct_result)
    
    # Convert to DataFrame
    pct_df = pd.DataFrame(results)
    
    # Save detailed results
    output_file = output_dir / 'episode_pct_change_detailed.xlsx'
    pct_df.to_excel(output_file, index=False)
    print(f"\\n💾 Saved: {output_file}")
    
    # Create pivot grid
    pct_grid = pct_df.pivot(index='EpisodeID', columns='TagName', values='pct_change')
    pct_grid.to_excel(output_dir / 'grid_pct_change_transition.xlsx')
    print(f"   Saved percentage change grid")
    
    # Summary stats
    summary = {
        'total_episodes': len(episodes_df),
        'total_tags': len(pv_cols),
        'most_changed_tags': pct_df.groupby('TagName')['pct_change'].apply(
            lambda x: np.nanmean(np.abs(x))
        ).nlargest(5).to_dict()
    }
    
    if args.output_json:
        with open(output_dir / 'pct_change_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    print(f"\\n✅ Analysis complete!")
    return summary


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
