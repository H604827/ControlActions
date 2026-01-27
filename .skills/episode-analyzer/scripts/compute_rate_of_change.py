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
from shared.episode_utils import (
    load_ssd_data,
    load_ground_truth_with_fallback,
    get_unique_episodes,
    find_lowest_1071_timestamp,
    compute_percentage_change
)


DEFAULT_TS_PATH = 'DATA/03LIC_1071_JAN_2026.parquet'
DEFAULT_OUTPUT_DIR = 'RESULTS'


def parse_args():
    parser = argparse.ArgumentParser(description='Compute percentage change metrics per episode')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-trip-filter', action='store_true', help='Disable trip filtering')
    parser.add_argument('--ssd-file', type=str, default='DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx')
    parser.add_argument('--ts-file', type=str, default=DEFAULT_TS_PATH)
    parser.add_argument('--ground-truth', type=str, default='DATA/Updated Ground truth -Adnoc RCA - recent(all_episode_top5_test_validated).csv',
                        help='Path to ground truth CSV with AlarmStart_rounded column for episode filtering')
    parser.add_argument('--no-ground-truth-filter', action='store_true',
                        help='Disable filtering episodes to those in ground truth file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--output-json', action='store_true')
    return parser.parse_args()


def run_analysis(args):
    print(f"📊 Percentage Change Analysis")
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
