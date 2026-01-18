#!/usr/bin/env python3
"""
Map relationships between different data sources in the Control Actions project.
Connects time series columns, event sources, and knowledge graph tags.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime


def strings_similar(s1: str, s2: str) -> bool:
    """
    Check if two tag names are similar (handles naming variations).
    First 3 characters must match exactly.
    Characters after underscore must match or have 4+ char common substring.
    """
    s1 = str(s1).strip().replace(' ', '').upper()
    s2 = str(s2).strip().replace(' ', '').upper()
    
    if len(s1) < 3 or len(s2) < 3:
        return False
    if s1[:3] != s2[:3]:
        return False
    
    if '_' not in s1 or '_' not in s2:
        return False
    
    s1_after = s1.split('_', 1)[1]
    s2_after = s2.split('_', 1)[1]
    
    if s1_after == s2_after:
        return True
    
    shorter = s1_after if len(s1_after) <= len(s2_after) else s2_after
    longer = s2_after if len(s1_after) <= len(s2_after) else s1_after
    
    for i in range(len(shorter) - 3):
        if shorter[i:i+4] in longer:
            return True
    return False


def load_all_data():
    """Load all data sources."""
    # Time series
    ts_df = pd.read_parquet('DATA/03LIC_1071_JAN_2026.parquet')
    if 'TimeStamp' in ts_df.columns:
        ts_df.set_index('TimeStamp', inplace=True)
    ts_df.sort_index(inplace=True)
    
    # Events
    events_df = pd.read_csv('DATA/df_df_events_1071_export.csv', low_memory=False)
    events_df['VT_Start'] = pd.to_datetime(events_df['VT_Start'])
    events_df = events_df.sort_values('VT_Start')
    
    # Knowledge graph tags
    kg_df = pd.read_csv('DATA/03LIC1071_PropaneLoop_0426.csv')
    
    return ts_df, events_df, kg_df


def map_event_sources_to_timeseries(ts_df: pd.DataFrame, events_df: pd.DataFrame) -> dict:
    """Map event Source values to time series columns."""
    # Get unique event sources
    event_sources = events_df['Source'].unique()
    
    # Get time series base tags (without .PV/.OP suffix)
    ts_cols = ts_df.columns.tolist()
    ts_base_tags = set()
    for col in ts_cols:
        if col.endswith('.PV'):
            ts_base_tags.add(col.replace('.PV', ''))
        elif col.endswith('.OP'):
            ts_base_tags.add(col.replace('.OP', ''))
    
    # Map each event source to time series tags
    source_to_ts = {}
    unmatched_sources = []
    
    for source in event_sources:
        matches = []
        for ts_tag in ts_base_tags:
            if strings_similar(source, ts_tag):
                matches.append(ts_tag)
        
        if matches:
            source_to_ts[source] = {
                'matched_tags': matches,
                'pv_columns': [f"{t}.PV" for t in matches if f"{t}.PV" in ts_cols],
                'op_columns': [f"{t}.OP" for t in matches if f"{t}.OP" in ts_cols]
            }
        else:
            unmatched_sources.append(source)
    
    return {
        'matched': source_to_ts,
        'unmatched': unmatched_sources,
        'match_rate': len(source_to_ts) / len(event_sources) * 100 if len(event_sources) > 0 else 0
    }


def map_kg_tags_to_sources(kg_df: pd.DataFrame, events_df: pd.DataFrame, ts_df: pd.DataFrame) -> dict:
    """Map knowledge graph tags to event sources and time series."""
    kg_tags = kg_df['tagName'].unique()
    event_sources = events_df['Source'].unique()
    
    ts_cols = ts_df.columns.tolist()
    ts_base_tags = set()
    for col in ts_cols:
        if col.endswith('.PV'):
            ts_base_tags.add(col.replace('.PV', ''))
        elif col.endswith('.OP'):
            ts_base_tags.add(col.replace('.OP', ''))
    
    kg_mapping = {}
    
    for kg_tag in kg_tags:
        # Find matching event sources
        matching_sources = [s for s in event_sources if strings_similar(kg_tag, s)]
        
        # Find matching time series tags
        matching_ts = [t for t in ts_base_tags if strings_similar(kg_tag, t)]
        
        kg_mapping[kg_tag] = {
            'event_sources': matching_sources,
            'timeseries_tags': matching_ts,
            'has_events': len(matching_sources) > 0,
            'has_timeseries': len(matching_ts) > 0,
            'has_pv': any(f"{t}.PV" in ts_cols for t in matching_ts),
            'has_op': any(f"{t}.OP" in ts_cols for t in matching_ts)
        }
    
    # Summary stats
    with_events = sum(1 for m in kg_mapping.values() if m['has_events'])
    with_ts = sum(1 for m in kg_mapping.values() if m['has_timeseries'])
    with_both = sum(1 for m in kg_mapping.values() if m['has_events'] and m['has_timeseries'])
    controllable = sum(1 for m in kg_mapping.values() if m['has_pv'] and m['has_op'])
    
    return {
        'mappings': kg_mapping,
        'summary': {
            'total_kg_tags': len(kg_tags),
            'with_events': with_events,
            'with_timeseries': with_ts,
            'with_both': with_both,
            'controllable': controllable
        }
    }


def analyze_change_events(events_df: pd.DataFrame) -> dict:
    """Analyze CHANGE events (operator actions)."""
    change_events = events_df[events_df['ConditionName'] == 'CHANGE'].copy()
    
    # Get unique sources with CHANGE events
    change_sources = change_events['Source'].unique()
    
    # Calculate action statistics per source
    source_stats = {}
    for source in change_sources:
        source_changes = change_events[change_events['Source'] == source]
        
        # Try to calculate magnitude
        try:
            magnitudes = pd.to_numeric(source_changes['Value'], errors='coerce') - \
                         pd.to_numeric(source_changes['PrevValue'], errors='coerce')
            magnitudes = magnitudes.dropna()
            
            if len(magnitudes) > 0:
                source_stats[source] = {
                    'count': len(source_changes),
                    'avg_magnitude': float(magnitudes.mean()),
                    'std_magnitude': float(magnitudes.std()),
                    'min_magnitude': float(magnitudes.min()),
                    'max_magnitude': float(magnitudes.max()),
                    'positive_changes': int((magnitudes > 0).sum()),
                    'negative_changes': int((magnitudes < 0).sum())
                }
            else:
                source_stats[source] = {'count': len(source_changes), 'magnitude_available': False}
        except Exception:
            source_stats[source] = {'count': len(source_changes), 'magnitude_available': False}
    
    return {
        'total_change_events': len(change_events),
        'unique_sources': len(change_sources),
        'time_range': {
            'start': str(change_events['VT_Start'].min()),
            'end': str(change_events['VT_Start'].max())
        },
        'source_statistics': source_stats
    }


def generate_relationship_report() -> dict:
    """Generate comprehensive relationship mapping report."""
    print("Loading data sources...")
    ts_df, events_df, kg_df = load_all_data()
    
    print("Mapping event sources to time series...")
    event_ts_mapping = map_event_sources_to_timeseries(ts_df, events_df)
    
    print("Mapping knowledge graph tags...")
    kg_mapping = map_kg_tags_to_sources(kg_df, events_df, ts_df)
    
    print("Analyzing CHANGE events...")
    change_analysis = analyze_change_events(events_df)
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'event_to_timeseries': event_ts_mapping,
        'knowledge_graph': kg_mapping,
        'change_events': change_analysis
    }
    
    return report


def print_summary(report: dict):
    """Print human-readable summary."""
    print("\n" + "="*60)
    print("DATA RELATIONSHIP MAPPING REPORT")
    print("="*60)
    
    # Event to time series mapping
    evt_ts = report['event_to_timeseries']
    print(f"\n📊 Event Sources → Time Series Mapping:")
    print(f"   Matched sources: {len(evt_ts['matched'])}")
    print(f"   Unmatched sources: {len(evt_ts['unmatched'])}")
    print(f"   Match rate: {evt_ts['match_rate']:.1f}%")
    
    # Knowledge graph summary
    kg = report['knowledge_graph']['summary']
    print(f"\n🔗 Knowledge Graph Tags:")
    print(f"   Total KG tags: {kg['total_kg_tags']}")
    print(f"   With event data: {kg['with_events']}")
    print(f"   With time series: {kg['with_timeseries']}")
    print(f"   With both: {kg['with_both']}")
    print(f"   Controllable (PV+OP): {kg['controllable']}")
    
    # CHANGE events
    change = report['change_events']
    print(f"\n⚙️  Operator Actions (CHANGE events):")
    print(f"   Total events: {change['total_change_events']:,}")
    print(f"   Unique sources: {change['unique_sources']}")
    
    # Top sources by action count
    top_sources = sorted(change['source_statistics'].items(), 
                         key=lambda x: x[1].get('count', 0), reverse=True)[:5]
    print(f"\n   Top 5 most operated tags:")
    for source, stats in top_sources:
        count = stats.get('count', 0)
        if stats.get('magnitude_available', True) and 'avg_magnitude' in stats:
            avg_mag = stats['avg_magnitude']
            print(f"   - {source}: {count} actions, avg magnitude: {avg_mag:+.2f}")
        else:
            print(f"   - {source}: {count} actions")


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Map relationships between data sources')
    parser.add_argument('--output', '-o', default='RESULTS/data_relationships.json',
                        help='Output JSON report path')
    args = parser.parse_args()
    
    report = generate_relationship_report()
    print_summary(report)
    
    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Full report saved to: {args.output}")


if __name__ == '__main__':
    main()
