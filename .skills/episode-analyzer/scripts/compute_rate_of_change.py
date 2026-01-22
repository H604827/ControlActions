#!/usr/bin/env python3
"""
Rate of Change Calculator for Alarm Episodes.

Computes rate of change metrics for each tag within each episode:
- Mean, max, min, std of rate of change
- Trend direction (increasing/decreasing/stable)
- Total change within the period

Usage:
    python .skills/episode-analyzer/scripts/compute_rate_of_change.py \
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
    parser = argparse.ArgumentParser(description='Compute rate of change metrics per episode')
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


def compute_roc_metrics(ts_df: pd.DataFrame, start_time: pd.Timestamp, 
                        end_time: pd.Timestamp, tag_col: str) -> dict:
    """Compute rate of change metrics for a tag within a time window."""
    mask = (ts_df.index >= start_time) & (ts_df.index <= end_time)
    window_df = ts_df.loc[mask, [tag_col]].dropna()
    
    if len(window_df) < 2:
        return {
            'mean_roc': np.nan, 'max_roc': np.nan, 'min_roc': np.nan,
            'std_roc': np.nan, 'abs_mean_roc': np.nan,
            'trend_direction': 'insufficient_data',
            'start_value': np.nan, 'end_value': np.nan,
            'total_change': np.nan, 'percent_change': np.nan,
            'data_points': len(window_df),
            'time_window_minutes': 0
        }
    
    values = window_df[tag_col].values
    times_minutes = (window_df.index - window_df.index[0]).total_seconds() / 60
    
    # Calculate derivatives
    time_diff = np.diff(times_minutes)
    time_diff[time_diff == 0] = 1/60
    value_diff = np.diff(values)
    roc = value_diff / time_diff
    
    start_val = values[0]
    end_val = values[-1]
    total_change = end_val - start_val
    pct_change = (total_change / abs(start_val) * 100) if start_val != 0 else np.nan
    
    # Determine trend
    if abs(total_change) < 0.01 * abs(start_val) if start_val != 0 else abs(total_change) < 0.01:
        trend = 'stable'
    elif total_change > 0:
        trend = 'increasing'
    else:
        trend = 'decreasing'
    
    return {
        'mean_roc': float(np.nanmean(roc)),
        'max_roc': float(np.nanmax(roc)),
        'min_roc': float(np.nanmin(roc)),
        'std_roc': float(np.nanstd(roc)),
        'abs_mean_roc': float(np.nanmean(np.abs(roc))),
        'trend_direction': trend,
        'start_value': float(start_val),
        'end_value': float(end_val),
        'total_change': float(total_change),
        'percent_change': float(pct_change) if not np.isnan(pct_change) else None,
        'data_points': len(window_df),
        'time_window_minutes': float(times_minutes[-1])
    }


def run_analysis(args):
    print(f"📊 Rate of Change Analysis")
    print(f"=" * 60)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📁 Loading data...")
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
    print(f"\n🔄 Computing rate of change metrics...")
    results = []
    
    for idx, episode in episodes_df.iterrows():
        ep_id = episode['EpisodeID']
        transition_start = episode['EarliestTransitionStart']
        alarm_start = episode['AlarmStart_rounded_minutes']
        alarm_end = episode['AlarmEnd_rounded_minutes']
        
        if idx % 50 == 0:
            print(f"   Episode {ep_id}/{len(episodes_df)}...")
        
        for tag_col in pv_cols:
            # Transition period
            roc_trans = compute_roc_metrics(ts_df, transition_start, alarm_start, tag_col)
            roc_trans.update({
                'EpisodeID': ep_id,
                'TagName': tag_col,
                'Period': 'transition',
                'PeriodStart': str(transition_start),
                'PeriodEnd': str(alarm_start)
            })
            results.append(roc_trans)
            
            # Alarm period
            roc_alarm = compute_roc_metrics(ts_df, alarm_start, alarm_end, tag_col)
            roc_alarm.update({
                'EpisodeID': ep_id,
                'TagName': tag_col,
                'Period': 'alarm',
                'PeriodStart': str(alarm_start),
                'PeriodEnd': str(alarm_end)
            })
            results.append(roc_alarm)
    
    # Convert to DataFrame
    roc_df = pd.DataFrame(results)
    
    # Save detailed results
    output_file = output_dir / 'episode_rate_of_change_detailed.xlsx'
    roc_df.to_excel(output_file, index=False)
    print(f"\n💾 Saved: {output_file}")
    
    # Create pivot grids
    for period in ['transition', 'alarm']:
        period_df = roc_df[roc_df['Period'] == period]
        
        # Mean ROC grid (numeric values)
        roc_grid = period_df.pivot(index='EpisodeID', columns='TagName', values='mean_roc')
        roc_grid.to_excel(output_dir / f'grid_roc_mean_{period}.xlsx')
    
    print(f"   Saved trend and ROC grids for both periods")
    
    # Summary stats
    summary = {
        'total_episodes': len(episodes_df),
        'total_tags': len(pv_cols),
        'transition_period': {
            'most_volatile_tags': roc_df[roc_df['Period'] == 'transition'].groupby('TagName')['std_roc'].mean().nlargest(5).to_dict(),
            'most_increasing_tags': roc_df[(roc_df['Period'] == 'transition') & (roc_df['trend_direction'] == 'increasing')].groupby('TagName').size().nlargest(5).to_dict()
        },
        'alarm_period': {
            'most_volatile_tags': roc_df[roc_df['Period'] == 'alarm'].groupby('TagName')['std_roc'].mean().nlargest(5).to_dict()
        }
    }
    
    if args.output_json:
        with open(output_dir / 'roc_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete!")
    return summary


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
