#!/usr/bin/env python3
"""
Build training features for ML models from operator actions.

This script creates a feature matrix that can be used to train models
to predict action magnitude and direction based on plant state.

Usage:
    python build_training_features.py \
        --events-file DATA/df_df_events_1071_export.csv \
        --ts-file DATA/03LIC_1071_JAN_2026.parquet \
        --target-tag 03LIC_1071 \
        --output-file RESULTS/training_features.csv
    
    # With trip filtering and date range:
    python build_training_features.py \
        --start-date 2025-01-01 --end-date 2025-06-30 \
        --output-file RESULTS/training_features_2025.csv
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.data_loader import load_all_data, filter_trip_periods, DataFilterStats


def get_pv_features_at_time(ts_df: pd.DataFrame, timestamp: pd.Timestamp,
                            window_minutes: int = 10) -> dict:
    """Extract PV features at a specific timestamp."""
    features = {}
    
    # Get data window before timestamp
    window_start = timestamp - pd.Timedelta(minutes=window_minutes)
    window_data = ts_df.loc[window_start:timestamp]
    
    if len(window_data) == 0:
        return features
    
    # Get PV columns
    pv_cols = [col for col in ts_df.columns if col.endswith('.PV')]
    
    for col in pv_cols:
        tag_base = col.replace('.PV', '')
        
        if col not in window_data.columns:
            continue
        
        series = window_data[col].dropna()
        
        if len(series) == 0:
            continue
        
        # Current value
        features[f'{tag_base}_PV_current'] = series.iloc[-1]
        
        # Rate of change (last 5 minutes)
        if len(series) >= 2:
            recent = series.tail(min(5, len(series)))
            rate = (recent.iloc[-1] - recent.iloc[0]) / len(recent)
            features[f'{tag_base}_PV_rate'] = rate
        
        # Volatility (std over window)
        if len(series) >= 3:
            features[f'{tag_base}_PV_volatility'] = series.std()
        
        # Min/Max in window
        features[f'{tag_base}_PV_min'] = series.min()
        features[f'{tag_base}_PV_max'] = series.max()
    
    return features


def get_target_distance_features(ts_df: pd.DataFrame, timestamp: pd.Timestamp,
                                 target_col: str, alarm_threshold: float) -> dict:
    """Get distance-to-alarm features for target tag."""
    features = {}
    
    try:
        window_start = timestamp - pd.Timedelta(minutes=10)
        window_data = ts_df.loc[window_start:timestamp, target_col].dropna()
        
        if len(window_data) == 0:
            return features
        
        current = window_data.iloc[-1]
        
        # Distance to alarm threshold
        features['distance_to_threshold'] = current - alarm_threshold
        features['pct_to_threshold'] = (current - alarm_threshold) / current * 100 if current != 0 else 0
        
        # Is already in alarm?
        features['is_below_threshold'] = 1 if current < alarm_threshold else 0
        
        # Time-weighted rate of approach to threshold
        if len(window_data) >= 2:
            rate = (window_data.iloc[-1] - window_data.iloc[0]) / len(window_data)
            # Negative rate means approaching threshold for PVLO
            features['approach_rate'] = -rate  # Positive = approaching alarm
            
            # Estimate time to alarm if rate continues
            if rate < 0:  # Moving toward alarm
                time_to_alarm = (current - alarm_threshold) / abs(rate)
                features['est_time_to_alarm_min'] = min(time_to_alarm, 999)
            else:
                features['est_time_to_alarm_min'] = 999
    
    except Exception as e:
        pass
    
    return features


def get_temporal_features(timestamp: pd.Timestamp) -> dict:
    """Extract temporal features."""
    return {
        'hour_of_day': timestamp.hour,
        'day_of_week': timestamp.dayofweek,
        'is_weekend': 1 if timestamp.dayofweek >= 5 else 0,
        'is_night_shift': 1 if timestamp.hour < 6 or timestamp.hour >= 22 else 0
    }


def get_recent_action_features(events_df: pd.DataFrame, timestamp: pd.Timestamp,
                               tag: str, lookback_minutes: int = 60) -> dict:
    """Get features about recent actions on this tag."""
    features = {}
    
    window_start = timestamp - pd.Timedelta(minutes=lookback_minutes)
    
    recent_actions = events_df[
        (events_df['ConditionName'] == 'CHANGE') &
        (events_df['Source'] == tag) &
        (events_df['VT_Start'] >= window_start) &
        (events_df['VT_Start'] < timestamp)
    ]
    
    features['recent_action_count'] = len(recent_actions)
    
    if len(recent_actions) > 0:
        # Time since last action
        last_action_time = recent_actions['VT_Start'].max()
        features['time_since_last_action_min'] = (timestamp - last_action_time).total_seconds() / 60
        
        # Last action magnitude
        last_action = recent_actions.loc[recent_actions['VT_Start'].idxmax()]
        val = pd.to_numeric(last_action['Value'], errors='coerce')
        prev = pd.to_numeric(last_action['PrevValue'], errors='coerce')
        if pd.notna(val) and pd.notna(prev):
            features['last_action_magnitude'] = val - prev
            features['last_action_direction'] = 1 if val > prev else -1 if val < prev else 0
    else:
        features['time_since_last_action_min'] = lookback_minutes
        features['last_action_magnitude'] = 0
        features['last_action_direction'] = 0
    
    return features


def build_training_dataset(events_df: pd.DataFrame, ts_df: pd.DataFrame,
                          target_tag: str = '03LIC_1071',
                          alarm_threshold: float = 28.75,
                          related_tags: List[str] = None) -> pd.DataFrame:
    """Build complete training dataset from CHANGE events."""
    
    target_pv_col = f'{target_tag}.PV'
    
    # Get CHANGE events
    change_events = events_df[events_df['ConditionName'] == 'CHANGE'].copy()
    
    # Calculate target variable (magnitude and direction)
    change_events['Value'] = pd.to_numeric(change_events['Value'], errors='coerce')
    change_events['PrevValue'] = pd.to_numeric(change_events['PrevValue'], errors='coerce')
    change_events['magnitude'] = change_events['Value'] - change_events['PrevValue']
    change_events['direction'] = np.where(change_events['magnitude'] > 0, 1, 
                                          np.where(change_events['magnitude'] < 0, -1, 0))
    
    # Filter valid events
    valid_events = change_events[change_events['magnitude'].notna()].copy()
    
    records = []
    
    for idx, event in valid_events.iterrows():
        timestamp = event['VT_Start']
        tag = event['Source']
        
        # Skip if timestamp not in time series range
        if timestamp < ts_df.index.min() or timestamp > ts_df.index.max():
            continue
        
        record = {
            'timestamp': timestamp,
            'action_tag': tag,
            'magnitude': event['magnitude'],
            'abs_magnitude': abs(event['magnitude']),
            'direction': event['direction'],
            'new_value': event['Value'],
            'prev_value': event['PrevValue']
        }
        
        # Add target distance features
        target_features = get_target_distance_features(ts_df, timestamp, target_pv_col, alarm_threshold)
        record.update(target_features)
        
        # Add target PV features
        if target_pv_col in ts_df.columns:
            pv_features = get_pv_features_at_time(ts_df, timestamp, window_minutes=10)
            # Only add target tag features to keep dataset manageable
            for key, val in pv_features.items():
                if target_tag in key:
                    record[key] = val
        
        # Add temporal features
        record.update(get_temporal_features(timestamp))
        
        # Add recent action features for this tag
        recent_features = get_recent_action_features(events_df, timestamp, tag)
        record.update(recent_features)
        
        records.append(record)
    
    return pd.DataFrame(records)


def summarize_dataset(df: pd.DataFrame) -> dict:
    """Summarize the training dataset."""
    if len(df) == 0:
        return {'error': 'Empty dataset'}
    
    summary = {
        'total_samples': len(df),
        'unique_tags': df['action_tag'].nunique(),
        'date_range': {
            'start': df['timestamp'].min().isoformat(),
            'end': df['timestamp'].max().isoformat()
        },
        'target_distribution': {
            'magnitude_mean': round(df['magnitude'].mean(), 4),
            'magnitude_std': round(df['magnitude'].std(), 4),
            'abs_magnitude_mean': round(df['abs_magnitude'].mean(), 4),
            'increase_pct': round((df['direction'] == 1).mean() * 100, 1),
            'decrease_pct': round((df['direction'] == -1).mean() * 100, 1)
        },
        'feature_count': len([c for c in df.columns if c not in ['timestamp', 'action_tag', 'magnitude', 'direction']]),
        'features': [c for c in df.columns if c not in ['timestamp', 'action_tag']]
    }
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Build training features from operator actions')
    parser.add_argument('--events-file', type=str, default='DATA/df_df_events_1071_export.csv',
                        help='Path to events CSV file')
    parser.add_argument('--ts-file', type=str, default='DATA/03LIC_1071_JAN_2026.parquet',
                        help='Path to time series parquet file')
    parser.add_argument('--trip-file', type=str, default='DATA/Final_List_Trip_Duration.csv',
                        help='Path to trip duration CSV file')
    parser.add_argument('--target-tag', type=str, default='03LIC_1071',
                        help='Target tag for alarm')
    parser.add_argument('--alarm-threshold', type=float, default=28.75,
                        help='Alarm threshold value')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output CSV file path')
    parser.add_argument('--output-json', action='store_true',
                        help='Output summary in JSON format')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-trip-filter', action='store_true',
                        help='Do not filter out trip periods')
    parser.add_argument('--recent', action='store_true',
                        help='Analyze only recent 6 months')
    parser.add_argument('--last-year', action='store_true',
                        help='Analyze only last year of data')
    
    args = parser.parse_args()
    
    # Determine date range
    start_date = args.start_date
    end_date = args.end_date
    
    if args.recent:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
        print(f"🕐 Analyzing recent data: {start_date} to {end_date}", file=sys.stderr)
    elif args.last_year:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        print(f"🕐 Analyzing last year: {start_date} to {end_date}", file=sys.stderr)
    
    # Load data using shared module
    print("Loading data...", file=sys.stderr)
    ts_df, events_df, stats = load_all_data(
        ts_path=args.ts_file,
        events_path=args.events_file,
        trip_path=args.trip_file if not args.no_trip_filter else None,
        start_date=start_date,
        end_date=end_date,
        filter_trips=not args.no_trip_filter,
        verbose=True
    )
    
    # Build dataset
    print("Building training features...", file=sys.stderr)
    training_df = build_training_dataset(
        events_df, ts_df,
        target_tag=args.target_tag,
        alarm_threshold=args.alarm_threshold
    )
    
    # Get summary
    summary = summarize_dataset(training_df)
    
    # Save output
    if args.output_file:
        training_df.to_csv(args.output_file, index=False)
        print(f"Training dataset saved to {args.output_file}", file=sys.stderr)
    
    if args.output_json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print("\n" + "=" * 60)
        print("TRAINING DATASET SUMMARY")
        print("=" * 60)
        
        print(f"\n📊 DATASET OVERVIEW")
        print(f"  Total samples: {summary['total_samples']}")
        print(f"  Unique tags: {summary['unique_tags']}")
        print(f"  Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"  Feature count: {summary['feature_count']}")
        
        print(f"\n🎯 TARGET VARIABLE DISTRIBUTION")
        td = summary['target_distribution']
        print(f"  Magnitude mean: {td['magnitude_mean']}")
        print(f"  Magnitude std: {td['magnitude_std']}")
        print(f"  Abs magnitude mean: {td['abs_magnitude_mean']}")
        print(f"  Increase actions: {td['increase_pct']}%")
        print(f"  Decrease actions: {td['decrease_pct']}%")
        
        print(f"\n📋 FEATURES")
        for f in summary['features'][:20]:
            print(f"  - {f}")
        if len(summary['features']) > 20:
            print(f"  ... and {len(summary['features'])-20} more")


if __name__ == '__main__':
    main()
