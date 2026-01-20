#!/usr/bin/env python3
"""
Estimate process response dynamics between OP and PV tags.
Determines how long it takes for control actions to affect process variables.

Improvements (Jan 2026):
- Trip period filtering to exclude abnormal plant states
- Date range filtering for recent data analysis
- Time-segmented analysis option
- Uses shared data loader for consistent preprocessing
"""

import sys
from pathlib import Path

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from scipy import signal
from scipy.optimize import curve_fit
import json
from datetime import datetime
from typing import Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Import shared utilities
from shared.data_loader import load_all_data, DataFilterStats
from shared.tag_utils import get_pv_op_pairs


def estimate_lag_crosscorr(pv_series: pd.Series, op_series: pd.Series, 
                           max_lag_minutes: int = 60) -> dict:
    """
    Estimate lag using cross-correlation.
    Returns optimal lag where PV best correlates with past OP changes.
    """
    # Align and clean data
    df = pd.DataFrame({'pv': pv_series, 'op': op_series}).dropna()
    
    if len(df) < 100:
        return {'error': 'Insufficient data', 'data_points': len(df)}
    
    pv = df['pv'].values
    op = df['op'].values
    
    # Normalize
    pv_norm = (pv - pv.mean()) / (pv.std() + 1e-10)
    op_norm = (op - op.mean()) / (op.std() + 1e-10)
    
    # Use OP derivative (changes) for better signal
    op_diff = np.diff(op_norm)
    pv_diff = np.diff(pv_norm)
    
    # Compute cross-correlation
    correlation = signal.correlate(pv_diff, op_diff, mode='full')
    lags = signal.correlation_lags(len(pv_diff), len(op_diff), mode='full')
    
    # Focus on positive lags (OP leads PV) within max_lag
    positive_mask = (lags >= 0) & (lags <= max_lag_minutes)
    lags_pos = lags[positive_mask]
    corr_pos = correlation[positive_mask]
    
    if len(corr_pos) == 0:
        return {'error': 'No valid lags found'}
    
    # Normalize correlation
    corr_normalized = corr_pos / (len(pv_diff) * np.std(pv_diff) * np.std(op_diff) + 1e-10)
    
    # Find peak
    peak_idx = np.argmax(np.abs(corr_normalized))
    optimal_lag = int(lags_pos[peak_idx])
    peak_corr = float(corr_normalized[peak_idx])
    
    # Find secondary peaks for complex dynamics
    peaks, properties = signal.find_peaks(np.abs(corr_normalized), height=0.1)
    secondary_lags = [int(lags_pos[p]) for p in peaks if p != peak_idx][:3]
    
    return {
        'optimal_lag_minutes': optimal_lag,
        'correlation_strength': peak_corr,
        'confidence': 'high' if abs(peak_corr) > 0.3 else 'medium' if abs(peak_corr) > 0.1 else 'low',
        'secondary_lags': secondary_lags,
        'data_points': len(df)
    }


def analyze_step_responses(ts_df: pd.DataFrame, events_df: pd.DataFrame, 
                           tag_name: str, window_before: int = 10, 
                           window_after: int = 30) -> dict:
    """
    Analyze PV responses after discrete OP CHANGE events.
    """
    pv_col = f'{tag_name}.PV'
    op_col = f'{tag_name}.OP'
    
    if pv_col not in ts_df.columns:
        return {'error': f'PV column {pv_col} not found'}
    
    # Get CHANGE events for this tag
    changes = events_df[
        (events_df['Source'] == tag_name) & 
        (events_df['ConditionName'] == 'CHANGE')
    ].copy()
    
    if len(changes) == 0:
        return {'error': f'No CHANGE events found for {tag_name}', 'event_count': 0}
    
    response_times = []
    settling_times = []
    valid_responses = 0
    
    for _, event in changes.iterrows():
        event_time = event['VT_Start']
        
        try:
            # Extract window around event
            start = event_time - pd.Timedelta(minutes=window_before)
            end = event_time + pd.Timedelta(minutes=window_after)
            
            window_data = ts_df.loc[start:end, pv_col].dropna()
            
            if len(window_data) < 10:
                continue
            
            # Split into before and after
            before_data = ts_df.loc[start:event_time, pv_col].dropna()
            after_data = ts_df.loc[event_time:end, pv_col].dropna()
            
            if len(before_data) < 3 or len(after_data) < 3:
                continue
            
            pv_before = before_data.mean()
            pv_final = after_data.iloc[-5:].mean() if len(after_data) >= 5 else after_data.mean()
            total_change = pv_final - pv_before
            
            if abs(total_change) < 0.01:  # No significant change
                continue
            
            # Find response time (when PV crosses 10% of total change)
            threshold_10 = pv_before + 0.1 * total_change
            for i, (ts, pv_val) in enumerate(after_data.items()):
                if (total_change > 0 and pv_val > threshold_10) or \
                   (total_change < 0 and pv_val < threshold_10):
                    response_time = (ts - event_time).total_seconds() / 60
                    response_times.append(response_time)
                    break
            
            # Find settling time (when PV stays within 5% of final)
            threshold_95 = pv_before + 0.95 * total_change
            for i, (ts, pv_val) in enumerate(after_data.items()):
                if (total_change > 0 and pv_val > threshold_95) or \
                   (total_change < 0 and pv_val < threshold_95):
                    settling_time = (ts - event_time).total_seconds() / 60
                    settling_times.append(settling_time)
                    break
            
            valid_responses += 1
            
        except Exception as e:
            continue
    
    result = {
        'tag': tag_name,
        'total_change_events': len(changes),
        'valid_responses_analyzed': valid_responses
    }
    
    if response_times:
        result['response_time'] = {
            'mean_minutes': float(np.mean(response_times)),
            'median_minutes': float(np.median(response_times)),
            'std_minutes': float(np.std(response_times)),
            'min_minutes': float(np.min(response_times)),
            'max_minutes': float(np.max(response_times)),
            'sample_count': len(response_times)
        }
    
    if settling_times:
        result['settling_time'] = {
            'mean_minutes': float(np.mean(settling_times)),
            'median_minutes': float(np.median(settling_times)),
            'std_minutes': float(np.std(settling_times)),
            'sample_count': len(settling_times)
        }
    
    return result


def estimate_time_constant(pv_series: pd.Series, op_series: pd.Series) -> dict:
    """
    Estimate first-order system time constant using ARX model approach.
    Model: PV[k] = a*PV[k-1] + b*OP[k-d] where d is dead time
    Time constant tau = -1/ln(a) (in sample periods)
    """
    df = pd.DataFrame({'pv': pv_series, 'op': op_series}).dropna()
    
    if len(df) < 100:
        return {'error': 'Insufficient data'}
    
    pv = df['pv'].values
    op = df['op'].values
    
    best_fit = {'r_squared': -np.inf}
    
    # Try different dead times
    for dead_time in range(0, 15):
        if dead_time >= len(pv) - 10:
            continue
            
        # Build regressor matrix
        y = pv[dead_time+1:]
        X = np.column_stack([
            pv[dead_time:-1],  # PV[k-1]
            op[:-dead_time-1] if dead_time > 0 else op[:-1]  # OP[k-d]
        ])
        
        if len(y) < 10:
            continue
        
        try:
            # Least squares fit
            coeffs, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
            a, b = coeffs
            
            # Calculate R-squared
            y_pred = X @ coeffs
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            
            if r_squared > best_fit['r_squared'] and 0 < a < 1:
                tau = -1 / np.log(a) if a > 0 else np.inf
                gain = b / (1 - a) if a != 1 else np.inf
                
                best_fit = {
                    'dead_time_minutes': dead_time,
                    'time_constant_minutes': float(tau),
                    'gain': float(gain),
                    'a_coefficient': float(a),
                    'b_coefficient': float(b),
                    'r_squared': float(r_squared)
                }
        except Exception:
            continue
    
    if best_fit['r_squared'] < 0:
        return {'error': 'Could not fit model'}
    
    return best_fit


def analyze_all_tags(ts_df: pd.DataFrame, events_df: pd.DataFrame) -> dict:
    """Analyze response dynamics for all PV/OP pairs."""
    pv_op_tags = get_pv_op_pairs(ts_df)
    
    results = {}
    
    for tag in pv_op_tags:
        print(f"  Analyzing {tag}...")
        
        pv_col = f'{tag}.PV'
        op_col = f'{tag}.OP'
        
        tag_result = {'tag': tag}
        
        # Cross-correlation
        xcorr = estimate_lag_crosscorr(ts_df[pv_col], ts_df[op_col])
        tag_result['cross_correlation'] = xcorr
        
        # Step response (if CHANGE events exist)
        step = analyze_step_responses(ts_df, events_df, tag)
        tag_result['step_response'] = step
        
        # Time constant estimation
        tc = estimate_time_constant(ts_df[pv_col], ts_df[op_col])
        tag_result['time_constant'] = tc
        
        results[tag] = tag_result
    
    return results


def generate_dynamics_report(ts_path: str = 'DATA/03LIC_1071_JAN_2026.parquet',
                             events_path: str = 'DATA/df_df_events_1071_export.csv',
                             trip_path: Optional[str] = 'DATA/Final_List_Trip_Duration.csv',
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             filter_trips: bool = True) -> dict:
    """
    Generate comprehensive response dynamics report.
    
    Args:
        ts_path: Path to time series data
        events_path: Path to events data
        trip_path: Path to trip duration data (None to skip filtering)
        start_date: Analyze data from this date (YYYY-MM-DD)
        end_date: Analyze data until this date (YYYY-MM-DD)
        filter_trips: Whether to exclude trip periods
    """
    print("Loading data...")
    
    # Use shared data loader for consistent preprocessing
    ts_df, events_df, stats = load_all_data(
        ts_path=ts_path,
        events_path=events_path,
        trip_path=trip_path if filter_trips else None,
        start_date=start_date,
        end_date=end_date,
        filter_trips=filter_trips,
        verbose=True  # Print filtering summary
    )
    
    print("\nAnalyzing all PV/OP pairs...")
    all_results = analyze_all_tags(ts_df, events_df)
    
    # Summary statistics
    lags = []
    time_constants = []
    
    for tag, result in all_results.items():
        if 'cross_correlation' in result and 'optimal_lag_minutes' in result['cross_correlation']:
            lags.append(result['cross_correlation']['optimal_lag_minutes'])
        if 'time_constant' in result and 'time_constant_minutes' in result['time_constant']:
            tc = result['time_constant']['time_constant_minutes']
            if tc < 100:  # Filter unreasonable values
                time_constants.append(tc)
    
    # Get actual date range of analyzed data
    data_start = ts_df.index.min()
    data_end = ts_df.index.max()
    
    summary = {
        'total_tags_analyzed': len(all_results),
        'data_range': {
            'start': str(data_start),
            'end': str(data_end),
            'days': (data_end - data_start).days
        },
        'filtering': {
            'trips_filtered': filter_trips,
            'date_range_applied': start_date is not None or end_date is not None,
            'start_date_filter': start_date,
            'end_date_filter': end_date,
            'rows_removed_trips': stats.ts_rows_in_trips if filter_trips else 0
        },
        'lag_statistics': {
            'mean_minutes': float(np.mean(lags)) if lags else None,
            'median_minutes': float(np.median(lags)) if lags else None,
            'min_minutes': float(np.min(lags)) if lags else None,
            'max_minutes': float(np.max(lags)) if lags else None
        },
        'time_constant_statistics': {
            'mean_minutes': float(np.mean(time_constants)) if time_constants else None,
            'median_minutes': float(np.median(time_constants)) if time_constants else None
        }
    }
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'summary': summary,
        'tag_results': all_results
    }
    
    return report


def print_summary(report: dict):
    """Print human-readable summary."""
    print("\n" + "="*60)
    print("RESPONSE DYNAMICS ESTIMATION REPORT")
    print("="*60)
    
    summary = report['summary']
    print(f"\n📊 Overview:")
    print(f"   Tags analyzed: {summary['total_tags_analyzed']}")
    
    # Show data range and filtering info
    if 'data_range' in summary:
        dr = summary['data_range']
        print(f"   Data range: {dr['start'][:10]} to {dr['end'][:10]} ({dr['days']} days)")
    
    if 'filtering' in summary:
        filt = summary['filtering']
        if filt['trips_filtered']:
            print(f"   Trip periods filtered: Yes ({filt['rows_removed_trips']:,} rows removed)")
        if filt['date_range_applied']:
            print(f"   Date filter: {filt['start_date_filter'] or 'start'} to {filt['end_date_filter'] or 'end'}")
    
    lag_stats = summary['lag_statistics']
    if lag_stats['mean_minutes'] is not None:
        print(f"\n⏱️  Lag Statistics (Cross-Correlation):")
        print(f"   Mean lag: {lag_stats['mean_minutes']:.1f} minutes")
        print(f"   Median lag: {lag_stats['median_minutes']:.1f} minutes")
        print(f"   Range: {lag_stats['min_minutes']:.0f} - {lag_stats['max_minutes']:.0f} minutes")
    
    tc_stats = summary['time_constant_statistics']
    if tc_stats['mean_minutes'] is not None:
        print(f"\n📈 Time Constant Statistics:")
        print(f"   Mean: {tc_stats['mean_minutes']:.1f} minutes")
        print(f"   Median: {tc_stats['median_minutes']:.1f} minutes")
    
    # Show top 5 fastest responding tags
    print(f"\n🚀 Fastest Responding Tags:")
    fast_tags = []
    for tag, result in report['tag_results'].items():
        if 'cross_correlation' in result and 'optimal_lag_minutes' in result['cross_correlation']:
            lag = result['cross_correlation']['optimal_lag_minutes']
            conf = result['cross_correlation'].get('confidence', 'unknown')
            fast_tags.append((tag, lag, conf))
    
    for tag, lag, conf in sorted(fast_tags, key=lambda x: x[1])[:5]:
        print(f"   - {tag}: {lag} min lag ({conf} confidence)")
    
    # Target tag detail
    target_tag = '03LIC_1071'
    if target_tag in report['tag_results']:
        print(f"\n🎯 Target Tag ({target_tag}) Details:")
        result = report['tag_results'][target_tag]
        
        if 'cross_correlation' in result:
            xcorr = result['cross_correlation']
            print(f"   Cross-correlation lag: {xcorr.get('optimal_lag_minutes', 'N/A')} min")
            print(f"   Confidence: {xcorr.get('confidence', 'N/A')}")
        
        if 'step_response' in result and 'response_time' in result['step_response']:
            rt = result['step_response']['response_time']
            print(f"   Step response time: {rt['mean_minutes']:.1f} min (mean)")
        
        if 'time_constant' in result and 'time_constant_minutes' in result['time_constant']:
            tc = result['time_constant']
            print(f"   Time constant: {tc['time_constant_minutes']:.1f} min")
            print(f"   Dead time: {tc['dead_time_minutes']} min")


def compare_time_periods(ts_path: str, events_path: str, trip_path: str,
                         periods: list, output_path: str = None) -> dict:
    """
    Compare response dynamics across different time periods.
    
    Args:
        ts_path: Path to time series data
        events_path: Path to events data
        trip_path: Path to trip duration data
        periods: List of dicts with 'name', 'start_date', 'end_date'
        output_path: Optional path to save comparison report
    
    Returns:
        Comparison report dict
    """
    print("=" * 60)
    print("TIME-SEGMENTED DYNAMICS COMPARISON")
    print("=" * 60)
    
    comparison = {
        'generated_at': datetime.now().isoformat(),
        'periods': {}
    }
    
    for period in periods:
        name = period['name']
        print(f"\n📅 Analyzing period: {name}")
        print(f"   Date range: {period['start_date']} to {period['end_date']}")
        
        report = generate_dynamics_report(
            ts_path=ts_path,
            events_path=events_path,
            trip_path=trip_path,
            start_date=period['start_date'],
            end_date=period['end_date'],
            filter_trips=True
        )
        
        comparison['periods'][name] = {
            'start_date': period['start_date'],
            'end_date': period['end_date'],
            'summary': report['summary'],
            'tag_results': report['tag_results']
        }
    
    # Generate comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    # Compare target tag across periods
    target_tag = '03LIC_1071'
    print(f"\n🎯 Target Tag ({target_tag}) Comparison:")
    print(f"{'Period':<20} {'Lag (min)':<12} {'Settling (min)':<15} {'Correlation':<12}")
    print("-" * 60)
    
    for name, data in comparison['periods'].items():
        if target_tag in data['tag_results']:
            result = data['tag_results'][target_tag]
            lag = result.get('cross_correlation', {}).get('optimal_lag_minutes', 'N/A')
            settling = result.get('step_response', {}).get('settling_time', {}).get('median_minutes', 'N/A')
            corr = result.get('cross_correlation', {}).get('correlation_strength', 'N/A')
            
            if isinstance(settling, float):
                settling = f"{settling:.1f}"
            if isinstance(corr, float):
                corr = f"{corr:.2f}"
            
            print(f"{name:<20} {str(lag):<12} {str(settling):<15} {str(corr):<12}")
    
    if output_path:
        output_p = Path(output_path)
        output_p.parent.mkdir(parents=True, exist_ok=True)
        with open(output_p, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        print(f"\n✅ Comparison report saved to: {output_path}")
    
    return comparison


def main():
    """Main entry point."""
    import argparse
    parser = argparse.ArgumentParser(description='Estimate process response dynamics')
    parser.add_argument('--ts-input', default='DATA/03LIC_1071_JAN_2026.parquet',
                        help='Time series parquet file')
    parser.add_argument('--events-input', default='DATA/df_df_events_1071_export.csv',
                        help='Events CSV file')
    parser.add_argument('--trip-input', default='DATA/Final_List_Trip_Duration.csv',
                        help='Trip duration CSV file')
    parser.add_argument('--output', '-o', default='RESULTS/response_dynamics.json',
                        help='Output JSON report path')
    parser.add_argument('--tag', '-t', default=None,
                        help='Analyze only this specific tag')
    parser.add_argument('--start-date', default=None,
                        help='Start date for analysis (YYYY-MM-DD). Use for recent data analysis.')
    parser.add_argument('--end-date', default=None,
                        help='End date for analysis (YYYY-MM-DD)')
    parser.add_argument('--no-trip-filter', action='store_true',
                        help='Disable trip period filtering')
    parser.add_argument('--recent', action='store_true',
                        help='Analyze only last 6 months of data (recommended)')
    parser.add_argument('--last-year', action='store_true',
                        help='Analyze only last 12 months of data')
    parser.add_argument('--compare-years', action='store_true',
                        help='Compare dynamics across years (2022, 2023, 2024, 2025)')
    args = parser.parse_args()
    
    # Handle year comparison mode
    if args.compare_years:
        periods = [
            {'name': '2022', 'start_date': '2022-01-01', 'end_date': '2022-12-31'},
            {'name': '2023', 'start_date': '2023-01-01', 'end_date': '2023-12-31'},
            {'name': '2024', 'start_date': '2024-01-01', 'end_date': '2024-12-31'},
            {'name': '2025 (YTD)', 'start_date': '2025-01-01', 'end_date': '2025-12-31'},
        ]
        compare_time_periods(
            args.ts_input, args.events_input, args.trip_input,
            periods, 'RESULTS/response_dynamics_yearly_comparison.json'
        )
        return
    
    # Handle convenience date range options
    start_date = args.start_date
    end_date = args.end_date
    
    if args.recent:
        # Last 6 months
        end_dt = datetime.now()
        start_dt = end_dt - pd.Timedelta(days=180)
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
        print(f"🕐 Analyzing recent data: {start_date} to {end_date}")
    elif args.last_year:
        # Last 12 months
        end_dt = datetime.now()
        start_dt = end_dt - pd.Timedelta(days=365)
        start_date = start_dt.strftime('%Y-%m-%d')
        end_date = end_dt.strftime('%Y-%m-%d')
        print(f"🕐 Analyzing last year: {start_date} to {end_date}")
    
    report = generate_dynamics_report(
        ts_path=args.ts_input,
        events_path=args.events_input,
        trip_path=args.trip_input if not args.no_trip_filter else None,
        start_date=start_date,
        end_date=end_date,
        filter_trips=not args.no_trip_filter
    )
    print_summary(report)
    
    # Save JSON report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n✅ Full report saved to: {args.output}")


if __name__ == '__main__':
    main()
