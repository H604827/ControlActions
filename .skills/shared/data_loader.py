#!/usr/bin/env python3
"""
Shared data loading and preprocessing utilities for Control Actions skills.

This module provides consistent data loading and filtering across all skills:
- Trip period filtering (removes data during plant shutdowns)
- Date range filtering (focus on recent data)
- Unified data loading with proper timestamp handling

Usage:
    from shared.data_loader import load_all_data, DataFilterStats
    
    ts_df, events_df, stats = load_all_data(
        filter_trips=True,
        start_date='2024-01-01',
        end_date='2025-06-30'
    )
"""

import pandas as pd
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
from datetime import datetime


@dataclass
class DataFilterStats:
    """Statistics about data filtering operations."""
    # Original counts
    original_ts_rows: int = 0
    original_events_rows: int = 0
    
    # After date filter
    ts_rows_after_date_filter: int = 0
    events_rows_after_date_filter: int = 0
    
    # Trip filtering
    ts_rows_in_trips: int = 0
    events_rows_in_trips: int = 0
    
    # Final counts
    final_ts_rows: int = 0
    final_events_rows: int = 0
    
    # Data range
    data_start: Optional[str] = None
    data_end: Optional[str] = None
    data_days: int = 0
    
    # Filter settings
    trips_filtered: bool = False
    start_date_filter: Optional[str] = None
    end_date_filter: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'original_ts_rows': self.original_ts_rows,
            'original_events_rows': self.original_events_rows,
            'ts_rows_after_date_filter': self.ts_rows_after_date_filter,
            'events_rows_after_date_filter': self.events_rows_after_date_filter,
            'ts_rows_in_trips': self.ts_rows_in_trips,
            'events_rows_in_trips': self.events_rows_in_trips,
            'final_ts_rows': self.final_ts_rows,
            'final_events_rows': self.final_events_rows,
            'data_start': self.data_start,
            'data_end': self.data_end,
            'data_days': self.data_days,
            'trips_filtered': self.trips_filtered,
            'start_date_filter': self.start_date_filter,
            'end_date_filter': self.end_date_filter
        }
    
    def print_summary(self):
        """Print a formatted summary of filtering stats."""
        print(f"\n📊 Data Filtering Summary:")
        print(f"   Original rows: {self.original_ts_rows:,} ts, {self.original_events_rows:,} events")
        
        if self.start_date_filter or self.end_date_filter:
            date_range = f"{self.start_date_filter or 'start'} to {self.end_date_filter or 'end'}"
            print(f"   Date range filter: {date_range}")
            print(f"   After date filter: {self.ts_rows_after_date_filter:,} ts, {self.events_rows_after_date_filter:,} events")
        
        if self.trips_filtered:
            print(f"   Rows removed (in trips): {self.ts_rows_in_trips:,} ts, {self.events_rows_in_trips:,} events")
        
        print(f"   Final rows for analysis: {self.final_ts_rows:,} ts, {self.final_events_rows:,} events")
        
        if self.data_start and self.data_end:
            print(f"   Data range: {self.data_start[:10]} to {self.data_end[:10]} ({self.data_days} days)")


# Default paths
DEFAULT_TS_PATH = 'DATA/03LIC_1071_JAN_2026.parquet'
DEFAULT_EVENTS_PATH = 'DATA/df_df_events_1071_export.csv'
DEFAULT_TRIP_PATH = 'DATA/Final_List_Trip_Duration.csv'
DEFAULT_KG_PATH = 'DATA/03LIC1071_PropaneLoop_0426.csv'


def load_trip_data(trip_path: str = DEFAULT_TRIP_PATH) -> pd.DataFrame:
    """
    Load trip duration data for filtering.
    
    Trip periods represent plant shutdowns/startups where:
    - 'Stop Date' = when plant stopped (trip started)
    - 'Start Date' = when plant restarted (trip ended)
    
    Data during these periods should be excluded from analysis as it
    represents abnormal operating conditions.
    """
    trip_df = pd.read_csv(trip_path)
    trip_df['Stop Date'] = pd.to_datetime(trip_df['Stop Date'])
    trip_df['Start Date'] = pd.to_datetime(trip_df['Start Date'])
    return trip_df


def filter_trip_periods(df: pd.DataFrame, trip_df: pd.DataFrame, 
                        time_col: Optional[str] = None) -> Tuple[pd.DataFrame, int]:
    """
    Filter out data points that fall within trip periods.
    
    Args:
        df: DataFrame with time index or time column
        trip_df: Trip duration DataFrame with 'Stop Date' and 'Start Date' columns
        time_col: Column name for time (if None, uses index)
    
    Returns:
        Tuple of (filtered DataFrame, count of removed rows)
    """
    if len(df) == 0:
        return df, 0
    
    if time_col:
        times = pd.to_datetime(df[time_col]).values
    else:
        times = df.index.values
    
    stop_dates = trip_df['Stop Date'].values
    start_dates = trip_df['Start Date'].values
    
    # Vectorized check: is each timestamp within any trip period?
    in_trip = np.zeros(len(times), dtype=bool)
    
    # Process in chunks to avoid memory issues with large arrays
    chunk_size = 100000
    for i in range(0, len(times), chunk_size):
        chunk = times[i:i+chunk_size]
        # Broadcasting: check if within any trip period
        # Trip period: from 'Stop Date' to 'Start Date'
        in_range = (chunk[:, None] >= stop_dates) & (chunk[:, None] <= start_dates)
        in_trip[i:i+chunk_size] = in_range.any(axis=1)
    
    removed_count = int(in_trip.sum())
    filtered_df = df[~in_trip].copy()
    
    return filtered_df, removed_count


def load_timeseries_data(ts_path: str = DEFAULT_TS_PATH,
                         trip_df: Optional[pd.DataFrame] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None,
                         filter_trips: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load and preprocess time series (PV/OP) data.
    
    Args:
        ts_path: Path to parquet file with time series data
        trip_df: Pre-loaded trip data (loads if None and filter_trips=True)
        start_date: Filter data after this date (YYYY-MM-DD)
        end_date: Filter data before this date (YYYY-MM-DD)
        filter_trips: Whether to filter out trip periods
    
    Returns:
        Tuple of (DataFrame, stats dict)
    """
    stats = {
        'original_rows': 0,
        'rows_after_date_filter': 0,
        'rows_in_trips': 0,
        'final_rows': 0
    }
    
    # Load data
    ts_df = pd.read_parquet(ts_path)
    if 'TimeStamp' in ts_df.columns:
        ts_df.set_index('TimeStamp', inplace=True)
    ts_df.sort_index(inplace=True)
    stats['original_rows'] = len(ts_df)
    
    # Apply date range filter
    if start_date:
        start_dt = pd.to_datetime(start_date)
        ts_df = ts_df[ts_df.index >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        ts_df = ts_df[ts_df.index <= end_dt]
    
    stats['rows_after_date_filter'] = len(ts_df)
    
    # Filter trip periods
    if filter_trips:
        if trip_df is None:
            trip_df = load_trip_data()
        ts_df, removed = filter_trip_periods(ts_df, trip_df)
        stats['rows_in_trips'] = removed
    
    stats['final_rows'] = len(ts_df)
    
    return ts_df, stats


def load_events_data(events_path: str = DEFAULT_EVENTS_PATH,
                     trip_df: Optional[pd.DataFrame] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None,
                     filter_trips: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load and preprocess events data.
    
    Note: The events CSV (df_df_events_1071_export.csv) already has correct
    timestamps and does NOT need the 1.5 hour offset adjustment that raw
    parquet files in events_1071/ folder require.
    
    Args:
        events_path: Path to events CSV file
        trip_df: Pre-loaded trip data (loads if None and filter_trips=True)
        start_date: Filter data after this date (YYYY-MM-DD)
        end_date: Filter data before this date (YYYY-MM-DD)
        filter_trips: Whether to filter out trip periods
    
    Returns:
        Tuple of (DataFrame, stats dict)
    """
    stats = {
        'original_rows': 0,
        'rows_after_date_filter': 0,
        'rows_in_trips': 0,
        'final_rows': 0
    }
    
    # Load data
    events_df = pd.read_csv(events_path, low_memory=False)
    events_df['VT_Start'] = pd.to_datetime(events_df['VT_Start'])
    events_df = events_df.sort_values('VT_Start')
    stats['original_rows'] = len(events_df)
    
    # Apply date range filter
    if start_date:
        start_dt = pd.to_datetime(start_date)
        events_df = events_df[events_df['VT_Start'] >= start_dt]
    
    if end_date:
        end_dt = pd.to_datetime(end_date)
        events_df = events_df[events_df['VT_Start'] <= end_dt]
    
    stats['rows_after_date_filter'] = len(events_df)
    
    # Filter trip periods
    if filter_trips:
        if trip_df is None:
            trip_df = load_trip_data()
        events_df, removed = filter_trip_periods(events_df, trip_df, time_col='VT_Start')
        stats['rows_in_trips'] = removed
    
    stats['final_rows'] = len(events_df)
    
    return events_df, stats


def load_all_data(ts_path: str = DEFAULT_TS_PATH,
                  events_path: str = DEFAULT_EVENTS_PATH,
                  trip_path: str = DEFAULT_TRIP_PATH,
                  start_date: Optional[str] = None,
                  end_date: Optional[str] = None,
                  filter_trips: bool = True,
                  verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, DataFilterStats]:
    """
    Load all data with consistent preprocessing.
    
    This is the main entry point for loading data in skills scripts.
    Ensures consistent filtering across time series and events data.
    
    Args:
        ts_path: Path to time series parquet file
        events_path: Path to events CSV file
        trip_path: Path to trip duration CSV file
        start_date: Filter data after this date (YYYY-MM-DD)
        end_date: Filter data before this date (YYYY-MM-DD)
        filter_trips: Whether to filter out trip periods
        verbose: Whether to print filtering summary
    
    Returns:
        Tuple of (ts_df, events_df, DataFilterStats)
    
    Example:
        ts_df, events_df, stats = load_all_data(
            start_date='2024-01-01',
            end_date='2025-06-30',
            filter_trips=True
        )
        stats.print_summary()
    """
    stats = DataFilterStats()
    stats.start_date_filter = start_date
    stats.end_date_filter = end_date
    stats.trips_filtered = filter_trips
    
    # Load trip data once (used for both ts and events)
    trip_df = None
    if filter_trips and trip_path is not None:
        trip_df = load_trip_data(trip_path)
    
    # Load time series (skip if path is None)
    ts_df = pd.DataFrame()
    if ts_path is not None:
        ts_df, ts_stats = load_timeseries_data(
            ts_path=ts_path,
            trip_df=trip_df,
            start_date=start_date,
            end_date=end_date,
            filter_trips=filter_trips
        )
        
        stats.original_ts_rows = ts_stats['original_rows']
        stats.ts_rows_after_date_filter = ts_stats['rows_after_date_filter']
        stats.ts_rows_in_trips = ts_stats['rows_in_trips']
        stats.final_ts_rows = ts_stats['final_rows']
    
    # Load events (skip if path is None)
    events_df = pd.DataFrame()
    if events_path is not None:
        events_df, events_stats = load_events_data(
            events_path=events_path,
            trip_df=trip_df,
            start_date=start_date,
            end_date=end_date,
            filter_trips=filter_trips
        )
        
        stats.original_events_rows = events_stats['original_rows']
        stats.events_rows_after_date_filter = events_stats['rows_after_date_filter']
        stats.events_rows_in_trips = events_stats['rows_in_trips']
        stats.final_events_rows = events_stats['final_rows']
    
    # Calculate data range
    if len(ts_df) > 0:
        stats.data_start = str(ts_df.index.min())
        stats.data_end = str(ts_df.index.max())
        stats.data_days = (ts_df.index.max() - ts_df.index.min()).days
    
    if verbose:
        stats.print_summary()
    
    return ts_df, events_df, stats


def load_kg_tags(kg_path: str = DEFAULT_KG_PATH) -> pd.DataFrame:
    """Load knowledge graph related tags."""
    return pd.read_csv(kg_path)


# Convenience functions for common date range patterns
def get_recent_date_range(months: int = 6) -> Tuple[str, str]:
    """Get date range for last N months."""
    end_dt = datetime.now()
    start_dt = end_dt - pd.Timedelta(days=30 * months)
    return start_dt.strftime('%Y-%m-%d'), end_dt.strftime('%Y-%m-%d')


def get_year_date_range(year: int) -> Tuple[str, str]:
    """Get date range for a specific year."""
    return f'{year}-01-01', f'{year}-12-31'


if __name__ == '__main__':
    # Quick test
    print("Testing shared data loader...")
    ts_df, events_df, stats = load_all_data(
        start_date='2025-01-01',
        end_date='2025-06-30',
        filter_trips=True
    )
    print(f"\nTS columns: {len(ts_df.columns)}")
    print(f"Sample TS data:\n{ts_df.head()}")
    print(f"\nEvents columns: {list(events_df.columns)[:5]}...")
