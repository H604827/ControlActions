#!/usr/bin/env python3
"""
Shared episode utilities for Control Actions skills.

Provides common functions for working with alarm episodes, SSD data,
operating limits, and ground truth data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Set, Tuple


# Default paths
DEFAULT_SSD_PATH = 'DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx'
DEFAULT_LIMITS_PATH = 'DATA/operating_limits.csv'
DEFAULT_GROUND_TRUTH_PATH = 'DATA/Updated Ground truth -Adnoc RCA - recent(all_episode_top5_test_validated).csv'


def load_ssd_data(ssd_path: str = DEFAULT_SSD_PATH, 
                  start_date: str = None, 
                  end_date: str = None) -> pd.DataFrame:
    """
    Load and preprocess SSD (Steady State Detection) data.
    
    Args:
        ssd_path: Path to SSD Excel file
        start_date: Filter data after this date (YYYY-MM-DD)
        end_date: Filter data before this date (YYYY-MM-DD)
    
    Returns:
        DataFrame with SSD episode data
    """
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


def load_operating_limits(limits_path: str = DEFAULT_LIMITS_PATH) -> pd.DataFrame:
    """
    Load operating limits data.
    
    Args:
        limits_path: Path to operating limits CSV file
    
    Returns:
        DataFrame indexed by TAG_NAME with LOWER_LIMIT and UPPER_LIMIT columns
    """
    limits_df = pd.read_csv(limits_path)
    return limits_df.set_index('TAG_NAME')


def load_ground_truth_alarm_starts(ground_truth_path: str = DEFAULT_GROUND_TRUTH_PATH) -> Set[pd.Timestamp]:
    """
    Load unique AlarmStart_rounded values from ground truth CSV.
    
    Args:
        ground_truth_path: Path to ground truth CSV file
    
    Returns:
        Set of datetime values for filtering episodes
    """
    gt_df = pd.read_csv(ground_truth_path)
    gt_df['AlarmStart_rounded'] = pd.to_datetime(gt_df['AlarmStart_rounded'])
    return set(gt_df['AlarmStart_rounded'].unique())


def get_unique_episodes(ssd_df: pd.DataFrame, 
                        ground_truth_alarm_starts: Set[pd.Timestamp] = None,
                        verbose: bool = True) -> pd.DataFrame:
    """
    Extract unique alarm episodes from SSD data.
    
    Args:
        ssd_df: SSD DataFrame with alarm episodes
        ground_truth_alarm_starts: Optional set of datetime values to filter episodes.
                                   If provided, only episodes with AlarmStart in this set are included.
        verbose: Whether to print filtering info
    
    Returns:
        DataFrame with unique episodes, each with:
        - EpisodeID: Sequential episode number
        - AlarmStart_rounded_minutes: Alarm start time
        - AlarmEnd_rounded_minutes: Alarm end time
        - EarliestTransitionStart: Earliest tag transition time
        - TransitionToAlarmMinutes: Duration from transition to alarm
        - AlarmDurationMinutes: Duration of alarm period
        - TotalEpisodeDurationMinutes: Total episode duration
    """
    # Group by alarm episode and get earliest transition start
    episodes = ssd_df.groupby(['AlarmStart_rounded_minutes', 'AlarmEnd_rounded_minutes']).agg({
        'Tag_First_Transition_Start_minutes': 'min'
    }).reset_index()
    
    episodes = episodes.rename(columns={
        'Tag_First_Transition_Start_minutes': 'EarliestTransitionStart'
    })
    
    # Filter to ground truth episodes if provided
    if ground_truth_alarm_starts is not None:
        original_count = len(episodes)
        episodes = episodes[episodes['AlarmStart_rounded_minutes'].isin(ground_truth_alarm_starts)]
        if verbose:
            print(f"   Filtered to ground truth episodes: {len(episodes)}/{original_count}")
    
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


def find_lowest_1071_timestamp(ts_df: pd.DataFrame, 
                                alarm_start: pd.Timestamp,
                                alarm_end: pd.Timestamp,
                                target_col: str = '03LIC_1071.PV') -> pd.Timestamp:
    """
    Find the timestamp where target tag PV is at its lowest during alarm period.
    
    Args:
        ts_df: Time series DataFrame with index as timestamp
        alarm_start: Start of alarm period
        alarm_end: End of alarm period
        target_col: Column name for target tag PV
    
    Returns:
        Timestamp of lowest PV value during alarm period
    """
    if target_col not in ts_df.columns:
        # Fallback to alarm_end if target tag not found
        return alarm_end
    
    mask = (ts_df.index >= alarm_start) & (ts_df.index <= alarm_end)
    alarm_window = ts_df.loc[mask, [target_col]].dropna()
    
    if len(alarm_window) == 0:
        return alarm_end
    
    # Find timestamp of minimum value
    return alarm_window[target_col].idxmin()


def compute_percentage_change(ts_df: pd.DataFrame, 
                               start_time: pd.Timestamp, 
                               end_time: pd.Timestamp, 
                               tag_col: str) -> dict:
    """
    Compute percentage change for a tag from start_time to end_time.
    
    Formula: (final_value - initial_value) / |initial_value| * 100
    
    Args:
        ts_df: Time series DataFrame with index as timestamp
        start_time: Start of analysis window
        end_time: End of analysis window
        tag_col: Column name for the tag
    
    Returns:
        Dictionary with percentage change metrics
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


def load_ground_truth_with_fallback(ground_truth_path: str = DEFAULT_GROUND_TRUTH_PATH,
                                     verbose: bool = True) -> Optional[Set[pd.Timestamp]]:
    """
    Load ground truth alarm starts with error handling and fallback.
    
    Args:
        ground_truth_path: Path to ground truth CSV file
        verbose: Whether to print status messages
    
    Returns:
        Set of datetime values, or None if loading fails
    """
    try:
        ground_truth_alarm_starts = load_ground_truth_alarm_starts(ground_truth_path)
        if verbose:
            print(f"   Ground truth alarm starts loaded: {len(ground_truth_alarm_starts)} unique episodes")
        return ground_truth_alarm_starts
    except FileNotFoundError:
        if verbose:
            print(f"   ⚠️ Ground truth file not found: {ground_truth_path}")
            print(f"   Proceeding without ground truth filtering...")
        return None
    except Exception as e:
        if verbose:
            print(f"   ⚠️ Error loading ground truth: {e}")
            print(f"   Proceeding without ground truth filtering...")
        return None


def get_tags_in_episode(ssd_df: pd.DataFrame, 
                        alarm_start: pd.Timestamp, 
                        alarm_end: pd.Timestamp) -> list:
    """
    Get list of tags that transitioned during an episode.
    
    Args:
        ssd_df: SSD DataFrame
        alarm_start: Alarm start time
        alarm_end: Alarm end time
    
    Returns:
        List of unique tag names in the episode
    """
    mask = (
        (ssd_df['AlarmStart_rounded_minutes'] == alarm_start) & 
        (ssd_df['AlarmEnd_rounded_minutes'] == alarm_end)
    )
    return ssd_df.loc[mask, 'TagName'].unique().tolist()


def get_unique_tags_from_ssd(ssd_df: pd.DataFrame) -> list:
    """
    Get unique tag base names from SSD data.
    
    Args:
        ssd_df: SSD DataFrame with TagName column
    
    Returns:
        Sorted list of base tag names (without .PV/.OP suffix)
    """
    tags = ssd_df['TagName'].unique()
    # Extract base name (remove .PV, .OP suffixes)
    base_tags = set()
    for tag in tags:
        base = str(tag).replace('.PV', '').replace('.OP', '')
        base_tags.add(base)
    return sorted(list(base_tags))


if __name__ == '__main__':
    # Quick test
    print("Testing episode utilities...")
    
    try:
        ssd_df = load_ssd_data()
        print(f"✅ SSD data loaded: {len(ssd_df)} rows")
        
        episodes = get_unique_episodes(ssd_df)
        print(f"✅ Episodes extracted: {len(episodes)}")
        
        limits_df = load_operating_limits()
        print(f"✅ Operating limits loaded: {len(limits_df)} tags")
        
    except FileNotFoundError as e:
        print(f"⚠️ Test skipped - file not found: {e}")
