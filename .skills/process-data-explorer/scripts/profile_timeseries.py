#!/usr/bin/env python3
"""
Profile PV/OP time series data for the Control Actions project.
Generates a comprehensive report of data structure, quality, and statistics.

Usage:
    python profile_timeseries.py --input DATA/03LIC_1071_JAN_2026.parquet
    python profile_timeseries.py --start-date 2025-01-01 --end-date 2025-06-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime, timedelta

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.data_loader import load_all_data, filter_trip_periods, DataFilterStats

def load_timeseries_data(filepath: str = 'DATA/03LIC_1071_JAN_2026.parquet') -> pd.DataFrame:
    """Load and prepare time series data."""
    df = pd.read_parquet(filepath)
    if 'TimeStamp' in df.columns:
        df.set_index('TimeStamp', inplace=True)
    df.sort_index(inplace=True)
    return df

def identify_tag_types(df: pd.DataFrame) -> dict:
    """Categorize columns into PV, OP, and other tags."""
    cols = df.columns.tolist()
    
    op_cols = [col for col in cols if col.endswith('.OP')]
    pv_cols = [col for col in cols if col.endswith('.PV')]
    other_cols = [col for col in cols if not col.endswith('.OP') and not col.endswith('.PV')]
    
    op_tags = {col.replace('.OP', '') for col in op_cols}
    pv_tags = {col.replace('.PV', '') for col in pv_cols}
    
    return {
        'op_columns': op_cols,
        'pv_columns': pv_cols,
        'other_columns': other_cols,
        'op_tags': list(op_tags),
        'pv_tags': list(pv_tags),
        'controllable_tags': list(op_tags & pv_tags),  # Have both PV and OP
        'op_only_tags': list(op_tags - pv_tags),
        'pv_only_tags': list(pv_tags - op_tags)
    }

def analyze_time_gaps(df: pd.DataFrame, threshold_minutes: int = 5) -> dict:
    """Analyze gaps in the time series."""
    if not isinstance(df.index, pd.DatetimeIndex):
        return {'error': 'Index is not DatetimeIndex'}
    
    time_diff = df.index.to_series().diff()
    gaps = time_diff[time_diff > pd.Timedelta(minutes=threshold_minutes)]
    
    gap_info = []
    for gap_end, duration in gaps.items():
        gap_start = gap_end - duration
        gap_info.append({
            'start': str(gap_start),
            'end': str(gap_end),
            'duration_minutes': duration.total_seconds() / 60
        })
    
    return {
        'total_gaps': len(gaps),
        'threshold_minutes': threshold_minutes,
        'largest_gap_minutes': float(gaps.max().total_seconds() / 60) if len(gaps) > 0 else 0,
        'gaps': gap_info[:20]  # First 20 gaps
    }

def analyze_column_statistics(df: pd.DataFrame) -> dict:
    """Generate statistics for each column."""
    stats = {}
    
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            stats[col] = {'status': 'all_null'}
            continue
        
        # Check if column is numeric
        if not np.issubdtype(df[col].dtype, np.number):
            # For non-numeric columns, provide value counts
            value_counts = df[col].value_counts()
            stats[col] = {
                'dtype': str(df[col].dtype),
                'count': int(len(col_data)),
                'missing': int(df[col].isnull().sum()),
                'missing_pct': float(df[col].isnull().sum() / len(df) * 100),
                'unique_values': int(df[col].nunique()),
                'top_values': value_counts.head(5).to_dict()
            }
            continue
            
        stats[col] = {
            'dtype': str(df[col].dtype),
            'count': int(len(col_data)),
            'missing': int(df[col].isnull().sum()),
            'missing_pct': float(df[col].isnull().sum() / len(df) * 100),
            'mean': float(col_data.mean()),
            'std': float(col_data.std()),
            'min': float(col_data.min()),
            'max': float(col_data.max()),
            'q25': float(col_data.quantile(0.25)),
            'q50': float(col_data.quantile(0.50)),
            'q75': float(col_data.quantile(0.75)),
            'iqr': float(col_data.quantile(0.75) - col_data.quantile(0.25))
        }
    
    return stats

def generate_profile_report(filepath: str = 'DATA/03LIC_1071_JAN_2026.parquet') -> dict:
    """Generate comprehensive profile report."""
    print(f"Loading data from {filepath}...")
    df = load_timeseries_data(filepath)
    
    print("Analyzing tag types...")
    tag_types = identify_tag_types(df)
    
    print("Analyzing time gaps...")
    gaps = analyze_time_gaps(df)
    
    print("Calculating column statistics...")
    col_stats = analyze_column_statistics(df)
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'source_file': filepath,
        'overview': {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'time_range_start': str(df.index.min()),
            'time_range_end': str(df.index.max()),
            'duration_days': (df.index.max() - df.index.min()).days
        },
        'tag_types': tag_types,
        'time_gaps': gaps,
        'column_statistics': col_stats
    }
    
    return report

def print_summary(report: dict):
    """Print human-readable summary."""
    print("\n" + "="*60)
    print("TIME SERIES DATA PROFILE REPORT")
    print("="*60)
    
    overview = report['overview']
    print(f"\n📊 Overview:")
    print(f"   Rows: {overview['total_rows']:,}")
    print(f"   Columns: {overview['total_columns']}")
    print(f"   Time Range: {overview['time_range_start'][:10]} to {overview['time_range_end'][:10]}")
    print(f"   Duration: {overview['duration_days']} days")
    
    tags = report['tag_types']
    print(f"\n🏷️  Tag Categories:")
    print(f"   Controllable (PV+OP): {len(tags['controllable_tags'])} tags")
    print(f"   PV only: {len(tags['pv_only_tags'])} tags")
    print(f"   OP only: {len(tags['op_only_tags'])} tags")
    
    print(f"\n📋 Controllable Tags (have both PV and OP):")
    for tag in sorted(tags['controllable_tags'])[:10]:
        print(f"   - {tag}")
    if len(tags['controllable_tags']) > 10:
        print(f"   ... and {len(tags['controllable_tags']) - 10} more")
    
    gaps = report['time_gaps']
    print(f"\n⏱️  Time Gaps (>{gaps['threshold_minutes']} min):")
    print(f"   Total gaps: {gaps['total_gaps']}")
    print(f"   Largest gap: {gaps['largest_gap_minutes']:.1f} minutes")
    
    # Check for columns with high missing values
    high_missing = [(col, stats['missing_pct']) 
                    for col, stats in report['column_statistics'].items() 
                    if isinstance(stats, dict) and 'missing_pct' in stats and stats['missing_pct'] > 5]
    
    if high_missing:
        print(f"\n⚠️  Columns with >5% missing values:")
        for col, pct in sorted(high_missing, key=lambda x: -x[1])[:5]:
            print(f"   - {col}: {pct:.1f}%")

def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Profile PV/OP time series data')
    parser.add_argument('--input', '-i', default='DATA/03LIC_1071_JAN_2026.parquet',
                        help='Input parquet file path')
    parser.add_argument('--trip-file', default='DATA/Final_List_Trip_Duration.csv',
                        help='Path to trip duration CSV file')
    parser.add_argument('--output', '-o', default='RESULTS/timeseries_profile.json',
                        help='Output JSON report path')
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
        print(f"🕐 Analyzing recent data: {start_date} to {end_date}")
    elif args.last_year:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        print(f"🕐 Analyzing last year: {start_date} to {end_date}")
    
    # Load data using shared module (only need ts_df)
    print("Loading data...")
    ts_df, _, stats = load_all_data(
        ts_path=args.input,
        events_path=None,
        trip_path=args.trip_file if not args.no_trip_filter else None,
        start_date=start_date,
        end_date=end_date,
        filter_trips=not args.no_trip_filter,
        verbose=True
    )
    
    print("Analyzing tag types...")
    tag_types = identify_tag_types(ts_df)
    
    print("Analyzing time gaps...")
    gaps = analyze_time_gaps(ts_df)
    
    print("Calculating column statistics...")
    col_stats = analyze_column_statistics(ts_df)
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'source_file': args.input,
        'filtering': {
            'trips_filtered': not args.no_trip_filter,
            'date_range_applied': start_date is not None or end_date is not None,
            'start_date_filter': start_date,
            'end_date_filter': end_date,
            'rows_removed_trips': stats.ts_rows_in_trips if not args.no_trip_filter else 0
        },
        'overview': {
            'total_rows': len(ts_df),
            'total_columns': len(ts_df.columns),
            'time_range_start': str(ts_df.index.min()),
            'time_range_end': str(ts_df.index.max()),
            'duration_days': (ts_df.index.max() - ts_df.index.min()).days
        },
        'tag_types': tag_types,
        'time_gaps': gaps,
        'column_statistics': col_stats
    }
    
    print_summary(report)
    
    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Full report saved to: {args.output}")

if __name__ == '__main__':
    main()
