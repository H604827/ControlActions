#!/usr/bin/env python3
"""
Analyze operator action magnitudes from CHANGE events.

This script extracts and analyzes the magnitudes (Value - PrevValue) of
operator actions to understand typical step sizes for each tag.

Usage:
    python analyze_action_magnitudes.py --events-file DATA/df_df_events_1071_export.csv
    python analyze_action_magnitudes.py --events-file DATA/df_df_events_1071_export.csv --output-json
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def load_events_data(events_file: str) -> pd.DataFrame:
    """Load and preprocess events data."""
    df = pd.read_csv(events_file, low_memory=False)
    df['VT_Start'] = pd.to_datetime(df['VT_Start'])
    df = df.sort_values('VT_Start')
    return df


def extract_change_events(events_df: pd.DataFrame) -> pd.DataFrame:
    """Extract CHANGE events and calculate magnitudes."""
    change_df = events_df[events_df['ConditionName'] == 'CHANGE'].copy()
    
    # Convert Value and PrevValue to numeric
    change_df['Value'] = pd.to_numeric(change_df['Value'], errors='coerce')
    change_df['PrevValue'] = pd.to_numeric(change_df['PrevValue'], errors='coerce')
    
    # Calculate magnitude
    change_df['magnitude'] = change_df['Value'] - change_df['PrevValue']
    
    # Calculate percentage change
    change_df['pct_change'] = np.where(
        change_df['PrevValue'] != 0,
        (change_df['magnitude'] / change_df['PrevValue']) * 100,
        np.nan
    )
    
    # Determine direction
    change_df['direction'] = np.where(
        change_df['magnitude'] > 0, 'increase',
        np.where(change_df['magnitude'] < 0, 'decrease', 'no_change')
    )
    
    return change_df


def analyze_magnitudes_by_tag(change_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate magnitude statistics per tag."""
    # Filter valid magnitudes
    valid_changes = change_df[change_df['magnitude'].notna()].copy()
    
    if len(valid_changes) == 0:
        return pd.DataFrame()
    
    # Group by Source (tag)
    stats = valid_changes.groupby('Source').agg({
        'magnitude': ['count', 'mean', 'median', 'std', 'min', 'max'],
        'pct_change': ['mean', 'median'],
        'direction': lambda x: (x == 'increase').sum()
    }).round(4)
    
    # Flatten column names
    stats.columns = [
        'action_count', 'magnitude_mean', 'magnitude_median', 'magnitude_std',
        'magnitude_min', 'magnitude_max', 'pct_change_mean', 'pct_change_median',
        'increase_count'
    ]
    
    # Calculate decrease count
    stats['decrease_count'] = stats['action_count'] - stats['increase_count']
    
    # Calculate increase ratio
    stats['increase_ratio'] = (stats['increase_count'] / stats['action_count']).round(3)
    
    # Add absolute magnitude stats (ignoring direction)
    abs_stats = valid_changes.groupby('Source')['magnitude'].apply(
        lambda x: x.abs().mean()
    )
    stats['abs_magnitude_mean'] = abs_stats.round(4)
    
    return stats.reset_index()


def get_typical_step_sizes(change_df: pd.DataFrame, percentiles=[25, 50, 75, 90]) -> dict:
    """Get typical step sizes across all tags."""
    valid_magnitudes = change_df['magnitude'].dropna()
    abs_magnitudes = valid_magnitudes.abs()
    
    result = {
        'total_actions': len(valid_magnitudes),
        'increases': (valid_magnitudes > 0).sum(),
        'decreases': (valid_magnitudes < 0).sum(),
        'no_change': (valid_magnitudes == 0).sum(),
        'overall_magnitude_stats': {
            'mean': round(valid_magnitudes.mean(), 4),
            'median': round(valid_magnitudes.median(), 4),
            'std': round(valid_magnitudes.std(), 4)
        },
        'absolute_magnitude_percentiles': {}
    }
    
    for p in percentiles:
        result['absolute_magnitude_percentiles'][f'p{p}'] = round(np.percentile(abs_magnitudes, p), 4)
    
    return result


def find_action_patterns(change_df: pd.DataFrame, min_actions: int = 5) -> list:
    """Find common action patterns per tag."""
    patterns = []
    
    for tag in change_df['Source'].unique():
        tag_actions = change_df[change_df['Source'] == tag]
        
        if len(tag_actions) < min_actions:
            continue
        
        valid_mags = tag_actions['magnitude'].dropna()
        if len(valid_mags) == 0:
            continue
        
        # Determine typical behavior
        increase_ratio = (valid_mags > 0).mean()
        
        pattern = {
            'tag': tag,
            'action_count': len(valid_mags),
            'typical_direction': 'increase' if increase_ratio > 0.6 else 'decrease' if increase_ratio < 0.4 else 'mixed',
            'increase_ratio': round(increase_ratio, 3),
            'typical_step_positive': round(valid_mags[valid_mags > 0].median(), 4) if (valid_mags > 0).any() else None,
            'typical_step_negative': round(valid_mags[valid_mags < 0].median(), 4) if (valid_mags < 0).any() else None,
            'step_consistency': round(1 - valid_mags.abs().std() / valid_mags.abs().mean(), 3) if valid_mags.abs().mean() > 0 else 0
        }
        
        patterns.append(pattern)
    
    # Sort by action count
    patterns.sort(key=lambda x: x['action_count'], reverse=True)
    
    return patterns


def main():
    parser = argparse.ArgumentParser(description='Analyze operator action magnitudes')
    parser.add_argument('--events-file', type=str, default='DATA/df_df_events_1071_export.csv',
                        help='Path to events CSV file')
    parser.add_argument('--output-json', action='store_true',
                        help='Output results in JSON format')
    parser.add_argument('--min-actions', type=int, default=3,
                        help='Minimum actions per tag to include in patterns')
    
    args = parser.parse_args()
    
    # Load data
    events_df = load_events_data(args.events_file)
    
    # Extract CHANGE events
    change_df = extract_change_events(events_df)
    
    # Analyze magnitudes by tag
    tag_stats = analyze_magnitudes_by_tag(change_df)
    
    # Get typical step sizes
    step_sizes = get_typical_step_sizes(change_df)
    
    # Find action patterns
    patterns = find_action_patterns(change_df, min_actions=args.min_actions)
    
    if args.output_json:
        result = {
            'summary': step_sizes,
            'per_tag_stats': tag_stats.to_dict('records') if not tag_stats.empty else [],
            'action_patterns': patterns
        }
        print(json.dumps(result, indent=2, default=str))
    else:
        print("=" * 60)
        print("OPERATOR ACTION MAGNITUDE ANALYSIS")
        print("=" * 60)
        
        print(f"\n📊 OVERALL SUMMARY")
        print(f"  Total CHANGE events: {step_sizes['total_actions']}")
        print(f"  Increases: {step_sizes['increases']} ({step_sizes['increases']/step_sizes['total_actions']*100:.1f}%)")
        print(f"  Decreases: {step_sizes['decreases']} ({step_sizes['decreases']/step_sizes['total_actions']*100:.1f}%)")
        
        print(f"\n📐 MAGNITUDE STATISTICS")
        print(f"  Mean: {step_sizes['overall_magnitude_stats']['mean']}")
        print(f"  Median: {step_sizes['overall_magnitude_stats']['median']}")
        print(f"  Std: {step_sizes['overall_magnitude_stats']['std']}")
        
        print(f"\n📏 TYPICAL STEP SIZES (absolute)")
        for k, v in step_sizes['absolute_magnitude_percentiles'].items():
            print(f"  {k}: {v}")
        
        print(f"\n🏷️  TOP TAGS BY ACTION COUNT")
        if not tag_stats.empty:
            top_tags = tag_stats.nlargest(10, 'action_count')
            for _, row in top_tags.iterrows():
                print(f"  {row['Source']}: {row['action_count']} actions, "
                      f"median={row['magnitude_median']:.3f}, "
                      f"increase_ratio={row['increase_ratio']:.2f}")
        
        print(f"\n🔄 ACTION PATTERNS (tags with {args.min_actions}+ actions)")
        for p in patterns[:10]:
            print(f"  {p['tag']}: {p['typical_direction']} tendency, "
                  f"{p['action_count']} actions, consistency={p['step_consistency']:.2f}")


if __name__ == '__main__':
    main()
