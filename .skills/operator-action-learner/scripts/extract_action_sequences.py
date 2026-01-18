#!/usr/bin/env python3
"""
Extract action sequences during alarm episodes.

This script identifies sequences of operator actions during each alarm episode
to understand action patterns, timing, and ordering.

Usage:
    python extract_action_sequences.py \
        --events-file DATA/df_df_events_1071_export.csv \
        --target-tag 03LIC_1071 \
        --alarm-type PVLO
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np


def load_events_data(events_file: str) -> pd.DataFrame:
    """Load and preprocess events data."""
    df = pd.read_csv(events_file, low_memory=False)
    df['VT_Start'] = pd.to_datetime(df['VT_Start'])
    df = df.sort_values('VT_Start')
    return df


def extract_alarm_episodes(events_df: pd.DataFrame, target_tag: str, 
                          alarm_type: str) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Extract alarm episodes as (start, end) tuples."""
    # Filter for target alarm events
    alarm_events = events_df[
        (events_df['Source'] == target_tag) &
        (events_df['ConditionName'] == alarm_type)
    ].copy()
    
    if 'Category' in alarm_events.columns:
        alarm_events = alarm_events[alarm_events['Category'] == 1]
    
    alarm_events = alarm_events.sort_values('VT_Start')
    
    episodes = []
    i = 0
    
    while i < len(alarm_events):
        row = alarm_events.iloc[i]
        action = row['Action']
        
        # Check if this is an alarm start (Action is null or blank)
        is_start = pd.isna(action) or action == ''
        
        if is_start:
            start_time = row['VT_Start']
            
            # Look for the corresponding end (Action == 'OK')
            remaining = alarm_events.iloc[i+1:]
            end_rows = remaining[remaining['Action'] == 'OK']
            
            if len(end_rows) > 0:
                end_time = end_rows.iloc[0]['VT_Start']
                episodes.append((start_time, end_time))
                
                # Move past this episode
                end_idx = alarm_events.index.get_loc(end_rows.index[0])
                i = end_idx + 1
            else:
                # No end found, skip
                i += 1
        else:
            i += 1
    
    return episodes


def get_actions_in_episode(events_df: pd.DataFrame, start: pd.Timestamp, 
                          end: pd.Timestamp, buffer_minutes: int = 30) -> pd.DataFrame:
    """Get CHANGE events during an alarm episode with optional buffer."""
    # Include buffer before alarm start (operators may act proactively)
    search_start = start - pd.Timedelta(minutes=buffer_minutes)
    
    change_events = events_df[
        (events_df['ConditionName'] == 'CHANGE') &
        (events_df['VT_Start'] >= search_start) &
        (events_df['VT_Start'] <= end)
    ].copy()
    
    if len(change_events) == 0:
        return pd.DataFrame()
    
    # Add timing info
    change_events['time_from_alarm_start'] = (
        change_events['VT_Start'] - start
    ).dt.total_seconds() / 60  # in minutes
    
    # Calculate magnitude
    change_events['Value'] = pd.to_numeric(change_events['Value'], errors='coerce')
    change_events['PrevValue'] = pd.to_numeric(change_events['PrevValue'], errors='coerce')
    change_events['magnitude'] = change_events['Value'] - change_events['PrevValue']
    
    return change_events.sort_values('VT_Start')


def analyze_episode_sequence(episode_actions: pd.DataFrame, start: pd.Timestamp,
                            end: pd.Timestamp) -> dict:
    """Analyze the action sequence within an episode."""
    if len(episode_actions) == 0:
        return {
            'action_count': 0,
            'sequence': [],
            'tags_operated': [],
            'first_action_delay_min': None,
            'inter_action_gap_mean_min': None
        }
    
    # Get unique tags in order of first appearance
    tags_order = episode_actions.drop_duplicates('Source')['Source'].tolist()
    
    # Build sequence
    sequence = []
    for _, row in episode_actions.iterrows():
        sequence.append({
            'tag': row['Source'],
            'time': row['VT_Start'].isoformat(),
            'time_from_start_min': round(row['time_from_alarm_start'], 2),
            'magnitude': round(row['magnitude'], 4) if pd.notna(row['magnitude']) else None,
            'direction': 'increase' if row.get('magnitude', 0) > 0 else 'decrease' if row.get('magnitude', 0) < 0 else 'unknown'
        })
    
    # Calculate inter-action gaps
    times = episode_actions['VT_Start'].tolist()
    gaps = [(times[i+1] - times[i]).total_seconds() / 60 for i in range(len(times)-1)]
    
    # First action delay (negative = before alarm, positive = after alarm)
    first_action = episode_actions.iloc[0]
    first_delay = (first_action['VT_Start'] - start).total_seconds() / 60
    
    return {
        'action_count': len(episode_actions),
        'sequence': sequence,
        'tags_operated': tags_order,
        'unique_tag_count': len(set(tags_order)),
        'first_action_delay_min': round(first_delay, 2),
        'first_action_tag': first_action['Source'],
        'inter_action_gap_mean_min': round(np.mean(gaps), 2) if gaps else None,
        'inter_action_gap_median_min': round(np.median(gaps), 2) if gaps else None,
        'total_duration_min': round((times[-1] - times[0]).total_seconds() / 60, 2) if len(times) > 1 else 0
    }


def extract_all_sequences(events_df: pd.DataFrame, episodes: List[Tuple],
                         buffer_minutes: int = 30) -> list:
    """Extract action sequences for all alarm episodes."""
    all_sequences = []
    
    for i, (start, end) in enumerate(episodes):
        episode_actions = get_actions_in_episode(events_df, start, end, buffer_minutes)
        analysis = analyze_episode_sequence(episode_actions, start, end)
        
        episode_duration = (end - start).total_seconds() / 60
        
        sequence_data = {
            'episode_id': i,
            'alarm_start': start.isoformat(),
            'alarm_end': end.isoformat(),
            'alarm_duration_min': round(episode_duration, 2),
            **analysis
        }
        
        all_sequences.append(sequence_data)
    
    return all_sequences


def find_common_patterns(sequences: list) -> dict:
    """Find common patterns across all sequences."""
    # Filter episodes with actions
    with_actions = [s for s in sequences if s['action_count'] > 0]
    
    if not with_actions:
        return {'no_patterns': True}
    
    # First action tag frequency
    first_tags = [s['first_action_tag'] for s in with_actions if s.get('first_action_tag')]
    first_tag_counts = pd.Series(first_tags).value_counts()
    
    # All tags operated frequency
    all_tags = []
    for s in with_actions:
        all_tags.extend(s.get('tags_operated', []))
    all_tag_counts = pd.Series(all_tags).value_counts()
    
    # Timing statistics
    first_delays = [s['first_action_delay_min'] for s in with_actions if s['first_action_delay_min'] is not None]
    action_counts = [s['action_count'] for s in with_actions]
    alarm_durations = [s['alarm_duration_min'] for s in with_actions]
    
    return {
        'episodes_total': len(sequences),
        'episodes_with_actions': len(with_actions),
        'episodes_without_actions': len(sequences) - len(with_actions),
        'first_action_tag_frequency': first_tag_counts.head(10).to_dict(),
        'all_tags_frequency': all_tag_counts.head(15).to_dict(),
        'timing_stats': {
            'first_action_delay_mean_min': round(np.mean(first_delays), 2) if first_delays else None,
            'first_action_delay_median_min': round(np.median(first_delays), 2) if first_delays else None,
            'actions_per_episode_mean': round(np.mean(action_counts), 2),
            'actions_per_episode_median': round(np.median(action_counts), 2),
            'alarm_duration_mean_min': round(np.mean(alarm_durations), 2),
            'alarm_duration_median_min': round(np.median(alarm_durations), 2)
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Extract action sequences during alarm episodes')
    parser.add_argument('--events-file', type=str, default='DATA/df_df_events_1071_export.csv',
                        help='Path to events CSV file')
    parser.add_argument('--target-tag', type=str, default='03LIC_1071',
                        help='Target tag for alarm episodes')
    parser.add_argument('--alarm-type', type=str, default='PVLO',
                        help='Alarm type (PVLO, PVHI, etc.)')
    parser.add_argument('--buffer-minutes', type=int, default=30,
                        help='Minutes before alarm to include in search')
    parser.add_argument('--output-json', action='store_true',
                        help='Output results in JSON format')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Output file path (JSON)')
    
    args = parser.parse_args()
    
    # Load data
    events_df = load_events_data(args.events_file)
    
    # Extract alarm episodes
    episodes = extract_alarm_episodes(events_df, args.target_tag, args.alarm_type)
    
    if not episodes:
        print(f"No alarm episodes found for {args.target_tag} {args.alarm_type}")
        return
    
    # Extract sequences
    sequences = extract_all_sequences(events_df, episodes, args.buffer_minutes)
    
    # Find common patterns
    patterns = find_common_patterns(sequences)
    
    result = {
        'target_tag': args.target_tag,
        'alarm_type': args.alarm_type,
        'patterns': patterns,
        'sequences': sequences
    }
    
    if args.output_file:
        with open(args.output_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"Results saved to {args.output_file}")
    elif args.output_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print("=" * 60)
        print(f"ACTION SEQUENCE ANALYSIS: {args.target_tag} {args.alarm_type}")
        print("=" * 60)
        
        print(f"\n📊 EPISODE SUMMARY")
        print(f"  Total alarm episodes: {patterns['episodes_total']}")
        print(f"  Episodes with actions: {patterns['episodes_with_actions']}")
        print(f"  Episodes without actions: {patterns['episodes_without_actions']}")
        
        if patterns.get('timing_stats'):
            ts = patterns['timing_stats']
            print(f"\n⏱️  TIMING STATISTICS")
            print(f"  Mean first action delay: {ts['first_action_delay_mean_min']} min")
            print(f"  Median first action delay: {ts['first_action_delay_median_min']} min")
            print(f"  Mean actions per episode: {ts['actions_per_episode_mean']}")
            print(f"  Mean alarm duration: {ts['alarm_duration_mean_min']} min")
        
        print(f"\n🏷️  FIRST ACTION TAG FREQUENCY")
        for tag, count in list(patterns.get('first_action_tag_frequency', {}).items())[:5]:
            print(f"  {tag}: {count} times")
        
        print(f"\n📋 ALL TAGS OPERATED (frequency)")
        for tag, count in list(patterns.get('all_tags_frequency', {}).items())[:10]:
            print(f"  {tag}: {count} times")
        
        print(f"\n📝 SAMPLE SEQUENCES (first 3 episodes with actions)")
        sample_count = 0
        for seq in sequences:
            if seq['action_count'] > 0 and sample_count < 3:
                print(f"\n  Episode {seq['episode_id']}: {seq['alarm_duration_min']:.1f} min duration")
                print(f"    First action: {seq['first_action_tag']} at {seq['first_action_delay_min']:.1f} min")
                print(f"    Actions: {seq['action_count']}, Tags: {seq['unique_tag_count']}")
                for action in seq['sequence'][:5]:
                    print(f"      {action['time_from_start_min']:+.1f}min: {action['tag']} {action['direction']} by {action['magnitude']}")
                if len(seq['sequence']) > 5:
                    print(f"      ... and {len(seq['sequence'])-5} more actions")
                sample_count += 1


if __name__ == '__main__':
    main()
