#!/usr/bin/env python3
"""
Visualize response dynamics for a specific PV/OP pair.
Creates plots showing cross-correlation and step responses.
"""

import pandas as pd
import numpy as np
from scipy import signal
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import argparse


def load_data(ts_path: str, events_path: str):
    """Load time series and events data."""
    ts_df = pd.read_parquet(ts_path)
    if 'TimeStamp' in ts_df.columns:
        ts_df.set_index('TimeStamp', inplace=True)
    ts_df.sort_index(inplace=True)
    
    events_df = pd.read_csv(events_path, low_memory=False)
    events_df['VT_Start'] = pd.to_datetime(events_df['VT_Start'])
    events_df = events_df.sort_values('VT_Start')
    
    return ts_df, events_df


def plot_cross_correlation(pv_series, op_series, tag_name, max_lag=60):
    """Create cross-correlation plot."""
    df = pd.DataFrame({'pv': pv_series, 'op': op_series}).dropna()
    
    pv = df['pv'].values
    op = df['op'].values
    
    # Use derivatives
    pv_diff = np.diff((pv - pv.mean()) / pv.std())
    op_diff = np.diff((op - op.mean()) / op.std())
    
    correlation = signal.correlate(pv_diff, op_diff, mode='full')
    lags = signal.correlation_lags(len(pv_diff), len(op_diff), mode='full')
    
    # Normalize
    corr_norm = correlation / (len(pv_diff) * np.std(pv_diff) * np.std(op_diff))
    
    # Focus on relevant range
    mask = (lags >= -10) & (lags <= max_lag)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=lags[mask],
        y=corr_norm[mask],
        mode='lines',
        name='Cross-correlation',
        line=dict(color='blue')
    ))
    
    # Mark peak
    pos_mask = (lags >= 0) & (lags <= max_lag)
    peak_idx = np.argmax(np.abs(corr_norm[pos_mask]))
    peak_lag = lags[pos_mask][peak_idx]
    peak_corr = corr_norm[pos_mask][peak_idx]
    
    fig.add_trace(go.Scatter(
        x=[peak_lag],
        y=[peak_corr],
        mode='markers',
        name=f'Peak at {peak_lag} min',
        marker=dict(size=12, color='red', symbol='star')
    ))
    
    fig.add_vline(x=0, line_dash='dash', line_color='gray')
    
    fig.update_layout(
        title=f'Cross-Correlation: {tag_name}.OP → {tag_name}.PV',
        xaxis_title='Lag (minutes)',
        yaxis_title='Correlation',
        showlegend=True,
        height=400
    )
    
    return fig


def plot_step_response_examples(ts_df, events_df, tag_name, 
                                 n_examples=5, window_before=10, window_after=30):
    """Plot example step responses around CHANGE events."""
    pv_col = f'{tag_name}.PV'
    op_col = f'{tag_name}.OP'
    
    changes = events_df[
        (events_df['Source'] == tag_name) & 
        (events_df['ConditionName'] == 'CHANGE')
    ].copy()
    
    if len(changes) == 0:
        return None
    
    # Sample some events
    sample_indices = np.random.choice(len(changes), min(n_examples, len(changes)), replace=False)
    sample_events = changes.iloc[sample_indices]
    
    fig = make_subplots(rows=n_examples, cols=1, 
                        shared_xaxes=True,
                        subplot_titles=[f"Event at {e['VT_Start']}" for _, e in sample_events.iterrows()])
    
    for i, (_, event) in enumerate(sample_events.iterrows(), 1):
        event_time = event['VT_Start']
        start = event_time - pd.Timedelta(minutes=window_before)
        end = event_time + pd.Timedelta(minutes=window_after)
        
        try:
            window_pv = ts_df.loc[start:end, pv_col]
            window_op = ts_df.loc[start:end, op_col]
            
            # Normalize time to minutes from event
            times = [(t - event_time).total_seconds() / 60 for t in window_pv.index]
            
            fig.add_trace(go.Scatter(
                x=times,
                y=window_pv.values,
                name='PV',
                line=dict(color='blue'),
                showlegend=(i == 1)
            ), row=i, col=1)
            
            fig.add_trace(go.Scatter(
                x=times,
                y=window_op.values,
                name='OP',
                line=dict(color='green', dash='dash'),
                yaxis='y2',
                showlegend=(i == 1)
            ), row=i, col=1)
            
            # Mark event time
            fig.add_vline(x=0, line_dash='dash', line_color='red', row=i, col=1)
            
        except Exception:
            continue
    
    fig.update_layout(
        title=f'Step Response Examples: {tag_name}',
        height=200 * n_examples,
        showlegend=True
    )
    fig.update_xaxes(title_text='Minutes from Event', row=n_examples, col=1)
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Visualize response dynamics')
    parser.add_argument('--tag', '-t', default='03LIC_1071',
                        help='Tag to analyze')
    parser.add_argument('--ts-input', default='DATA/03LIC_1071_JAN_2026.parquet',
                        help='Time series parquet file')
    parser.add_argument('--events-input', default='DATA/df_df_events_1071_export.csv',
                        help='Events CSV file')
    parser.add_argument('--output-dir', '-o', default='RESULTS/response-dynamics-estimator/dynamics_plots',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    print(f"Loading data...")
    ts_df, events_df = load_data(args.ts_input, args.events_input)
    
    pv_col = f'{args.tag}.PV'
    op_col = f'{args.tag}.OP'
    
    if pv_col not in ts_df.columns or op_col not in ts_df.columns:
        print(f"Error: Tag {args.tag} not found in data")
        return
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating cross-correlation plot...")
    xcorr_fig = plot_cross_correlation(ts_df[pv_col], ts_df[op_col], args.tag)
    xcorr_fig.write_html(output_dir / f'{args.tag}_crosscorr.html')
    print(f"  Saved: {output_dir / f'{args.tag}_crosscorr.html'}")
    
    print(f"Creating step response plots...")
    step_fig = plot_step_response_examples(ts_df, events_df, args.tag)
    if step_fig:
        step_fig.write_html(output_dir / f'{args.tag}_step_response.html')
        print(f"  Saved: {output_dir / f'{args.tag}_step_response.html'}")
    else:
        print(f"  No CHANGE events found for {args.tag}")
    
    print(f"\n✅ Plots saved to: {output_dir}")


if __name__ == '__main__':
    main()
