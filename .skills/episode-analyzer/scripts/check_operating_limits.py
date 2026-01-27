#!/usr/bin/env python3
"""
Operating Limit Deviation Checker for Alarm Episodes.

Checks if tags deviated from their operating limits during the 
transition period (Tag_First_Transition_Start to AlarmStart).

Uses DATA/operating_limits.csv for limit definitions.

Usage:
    python .skills/episode-analyzer/scripts/check_operating_limits.py \
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
    load_operating_limits,
    load_ground_truth_with_fallback,
    get_unique_episodes
)


DEFAULT_TS_PATH = 'DATA/03LIC_1071_JAN_2026.parquet'
DEFAULT_OUTPUT_DIR = 'RESULTS'


def parse_args():
    parser = argparse.ArgumentParser(description='Check operating limit deviations per episode')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-trip-filter', action='store_true', help='Disable trip filtering')
    parser.add_argument('--ssd-file', type=str, default='DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx')
    parser.add_argument('--ts-file', type=str, default=DEFAULT_TS_PATH)
    parser.add_argument('--operating-limits', type=str, default='DATA/operating_limits.csv')
    parser.add_argument('--ground-truth', type=str, default='DATA/Updated Ground truth -Adnoc RCA - recent(all_episode_top5_test_validated).csv',
                        help='Path to ground truth CSV with AlarmStart_rounded column for episode filtering')
    parser.add_argument('--no-ground-truth-filter', action='store_true',
                        help='Disable filtering episodes to those in ground truth file')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--output-json', action='store_true')
    return parser.parse_args()


def check_deviation(ts_df: pd.DataFrame, start_time: pd.Timestamp,
                    end_time: pd.Timestamp, tag_col: str, 
                    limits_df: pd.DataFrame) -> dict:
    """Check if a tag deviated from operating limits within a time window."""
    
    mask = (ts_df.index >= start_time) & (ts_df.index <= end_time)
    window_df = ts_df.loc[mask, [tag_col]].dropna()
    
    base_result = {
        'has_limits': False,
        'deviated': False,
        'deviation_type': 'no_data',
        'lower_limit': np.nan,
        'upper_limit': np.nan,
        'min_value': np.nan,
        'max_value': np.nan,
        'mean_value': np.nan,
        'first_deviation_time': None,
        'deviation_duration_minutes': 0.0,
        'max_deviation_pct': 0.0,
        'limit_breached': None,
        'below_lower_count': 0,
        'above_upper_count': 0,
        'data_points': len(window_df)
    }
    
    if len(window_df) == 0:
        return base_result
    
    values = window_df[tag_col].values
    times = window_df.index
    
    base_result['min_value'] = float(np.nanmin(values))
    base_result['max_value'] = float(np.nanmax(values))
    base_result['mean_value'] = float(np.nanmean(values))
    
    # Check if limits exist
    if tag_col not in limits_df.index:
        base_result['deviation_type'] = 'no_limits_defined'
        return base_result
    
    lower_limit = limits_df.loc[tag_col, 'LOWER_LIMIT']
    upper_limit = limits_df.loc[tag_col, 'UPPER_LIMIT']
    
    base_result['has_limits'] = True
    base_result['lower_limit'] = float(lower_limit)
    base_result['upper_limit'] = float(upper_limit)
    
    # Check for deviations
    below_lower = values < lower_limit
    above_upper = values > upper_limit
    deviated = below_lower | above_upper
    
    base_result['below_lower_count'] = int(below_lower.sum())
    base_result['above_upper_count'] = int(above_upper.sum())
    
    if not deviated.any():
        base_result['deviation_type'] = 'within_limits'
        return base_result
    
    base_result['deviated'] = True
    base_result['deviation_type'] = 'outside_limits'
    
    # First deviation time
    first_dev_idx = np.argmax(deviated)
    base_result['first_deviation_time'] = str(times[first_dev_idx])
    
    # Deviation duration (approximate based on ratio)
    deviation_ratio = deviated.sum() / len(values)
    total_minutes = (times[-1] - times[0]).total_seconds() / 60 if len(times) > 1 else 0
    base_result['deviation_duration_minutes'] = float(total_minutes * deviation_ratio)
    
    # Max deviation percentage
    limit_range = upper_limit - lower_limit
    if limit_range > 0:
        max_below = np.max(lower_limit - values[below_lower]) if below_lower.any() else 0
        max_above = np.max(values[above_upper] - upper_limit) if above_upper.any() else 0
        base_result['max_deviation_pct'] = float(max(max_below, max_above) / limit_range * 100)
    
    # Which limit breached
    if below_lower.any() and above_upper.any():
        base_result['limit_breached'] = 'both'
    elif below_lower.any():
        base_result['limit_breached'] = 'lower'
    else:
        base_result['limit_breached'] = 'upper'
    
    return base_result


def run_analysis(args):
    print(f"📊 Operating Limit Deviation Analysis")
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
    
    limits_df = load_operating_limits(args.operating_limits)
    print(f"   Operating limits defined for: {len(limits_df)} tags")
    
    pv_cols = [col for col in ts_df.columns if col.endswith('.PV')]
    pv_with_limits = [col for col in pv_cols if col in limits_df.index]
    print(f"   PV tags: {len(pv_cols)} (with limits: {len(pv_with_limits)})")
    
    # Check deviations
    print(f"\n🔄 Checking operating limit deviations...")
    results = []
    
    for idx, episode in episodes_df.iterrows():
        ep_id = episode['EpisodeID']
        transition_start = episode['EarliestTransitionStart']
        alarm_start = episode['AlarmStart_rounded_minutes']
        
        if idx % 50 == 0:
            print(f"   Episode {ep_id}/{len(episodes_df)}...")
        
        for tag_col in pv_cols:
            deviation = check_deviation(ts_df, transition_start, alarm_start, tag_col, limits_df)
            deviation.update({
                'EpisodeID': ep_id,
                'TagName': tag_col,
                'TransitionStart': str(transition_start),
                'AlarmStart': str(alarm_start)
            })
            results.append(deviation)
    
    # Convert to DataFrame
    dev_df = pd.DataFrame(results)
    
    # Save detailed results
    output_file = output_dir / 'episode_operating_limit_deviations_detailed.xlsx'
    dev_df.to_excel(output_file, index=False)
    print(f"\n💾 Saved: {output_file}")
    
    # Create deviation grid (binary: deviated or not)
    dev_grid = dev_df.pivot(index='EpisodeID', columns='TagName', values='deviated')
    dev_grid.to_excel(output_dir / 'grid_operating_limit_deviation.xlsx')
    print(f"   Saved deviation grid")
    
    # Create limit breached grid
    breach_grid = dev_df.pivot(index='EpisodeID', columns='TagName', values='limit_breached')
    breach_grid.to_excel(output_dir / 'grid_limit_breach_type.xlsx')
    print(f"   Saved breach type grid")
    
    # Summary statistics
    deviations_by_tag = dev_df[dev_df['deviated']].groupby('TagName').size()
    episodes_with_deviation = dev_df.groupby('EpisodeID')['deviated'].any()
    
    summary = {
        'total_episodes': len(episodes_df),
        'total_tags_checked': len(pv_cols),
        'tags_with_limits': len(pv_with_limits),
        'episodes_with_any_deviation': int(episodes_with_deviation.sum()),
        'episodes_without_deviation': int((~episodes_with_deviation).sum()),
        'most_deviating_tags': deviations_by_tag.nlargest(10).to_dict(),
        'deviation_counts': {
            'lower_limit_breaches': int((dev_df['limit_breached'] == 'lower').sum()),
            'upper_limit_breaches': int((dev_df['limit_breached'] == 'upper').sum()),
            'both_limits_breached': int((dev_df['limit_breached'] == 'both').sum())
        }
    }
    
    if args.output_json:
        with open(output_dir / 'operating_limit_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    print(f"\n✅ Analysis complete!")
    print(f"\n📈 Summary:")
    print(f"   Episodes with limit deviations: {summary['episodes_with_any_deviation']}/{summary['total_episodes']}")
    print(f"   Lower limit breaches: {summary['deviation_counts']['lower_limit_breaches']}")
    print(f"   Upper limit breaches: {summary['deviation_counts']['upper_limit_breaches']}")
    
    if summary['most_deviating_tags']:
        print(f"\n   Top deviating tags:")
        for tag, count in list(summary['most_deviating_tags'].items())[:5]:
            print(f"     {tag}: {count} episodes")
    
    return summary


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
