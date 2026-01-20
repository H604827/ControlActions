#!/usr/bin/env python3
"""
Map relationships between different data sources in the Control Actions project.
Connects time series columns, event sources, and knowledge graph tags.

Usage:
    python map_data_relationships.py
    python map_data_relationships.py --start-date 2025-01-01 --end-date 2025-06-30
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
from datetime import datetime, timedelta

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.data_loader import load_all_data as shared_load_all_data, DataFilterStats
from shared.tag_utils import strings_similar, get_pv_op_pairs, categorize_tags


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


def generate_relationship_report(ts_df: pd.DataFrame, events_df: pd.DataFrame) -> dict:
    """Generate comprehensive relationship mapping report."""
    # Load knowledge graph tags separately (not affected by trip/date filtering)
    kg_df = pd.read_csv('DATA/03LIC1071_PropaneLoop_0426.csv')
    
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
    parser.add_argument('--ts-file', default='DATA/03LIC_1071_JAN_2026.parquet',
                        help='Path to time series parquet file')
    parser.add_argument('--events-file', default='DATA/df_df_events_1071_export.csv',
                        help='Path to events CSV file')
    parser.add_argument('--trip-file', default='DATA/Final_List_Trip_Duration.csv',
                        help='Path to trip duration CSV file')
    parser.add_argument('--output', '-o', default='RESULTS/data_relationships.json',
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
    
    # Load data using shared module
    print("Loading data sources...")
    ts_df, events_df, stats = shared_load_all_data(
        ts_path=args.ts_file,
        events_path=args.events_file,
        trip_path=args.trip_file if not args.no_trip_filter else None,
        start_date=start_date,
        end_date=end_date,
        filter_trips=not args.no_trip_filter,
        verbose=True
    )
    
    report = generate_relationship_report(ts_df, events_df)
    
    # Add filtering info to report
    report['filtering'] = {
        'trips_filtered': not args.no_trip_filter,
        'date_range_applied': start_date is not None or end_date is not None,
        'start_date_filter': start_date,
        'end_date_filter': end_date,
        'rows_removed_trips_ts': stats.ts_rows_in_trips if not args.no_trip_filter else 0,
        'rows_removed_trips_events': stats.events_rows_in_trips if not args.no_trip_filter else 0
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
