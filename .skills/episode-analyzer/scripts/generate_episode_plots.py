#!/usr/bin/env python3
"""
Episode Visualizer - Generate Plotly visualizations for alarm episodes.

Generates interactive HTML visualizations showing:
- Target tag PV over time
- Alarm period shading
- Related tag PV/OP traces
- Operator action markers with hover details
- Operating limit lines

Usage:
    python .skills/episode-analyzer/scripts/generate_episode_plots.py \
        --start-date 2025-01-01 --end-date 2025-06-30 \
        --output-dir RESULTS/episode_plots
"""

import argparse
import sys
from pathlib import Path
from datetime import timedelta

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add shared module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.data_loader import load_all_data


# Default paths
DEFAULT_SSD_PATH = 'DATA/SSD_1071_SSD_output_1071_7Jan2026.xlsx'
DEFAULT_TS_PATH = 'DATA/03LIC_1071_JAN_2026.parquet'
DEFAULT_EVENTS_PATH = 'DATA/df_df_events_1071_export.csv'
DEFAULT_LIMITS_PATH = 'DATA/operating_limits.csv'
DEFAULT_OUTPUT_DIR = 'RESULTS/episode_plots'

# Target tag
TARGET_TAG = '03LIC_1071'
ALARM_THRESHOLD = 28.75


def parse_args():
    parser = argparse.ArgumentParser(description='Generate episode visualizations')
    parser.add_argument('--start-date', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--no-trip-filter', action='store_true', help='Disable trip filtering')
    parser.add_argument('--ssd-file', type=str, default=DEFAULT_SSD_PATH, help='Path to SSD file')
    parser.add_argument('--ts-file', type=str, default=DEFAULT_TS_PATH, help='Path to time series file')
    parser.add_argument('--events-file', type=str, default=DEFAULT_EVENTS_PATH, help='Path to events file')
    parser.add_argument('--operating-limits', type=str, default=DEFAULT_LIMITS_PATH, help='Path to operating limits')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR, help='Output directory for plots')
    parser.add_argument('--max-episodes', type=int, default=None, help='Maximum episodes to plot')
    parser.add_argument('--episode-ids', type=str, help='Comma-separated episode IDs to plot')
    parser.add_argument('--context-minutes', type=int, default=30, help='Minutes before/after episode to show')
    return parser.parse_args()


def load_ssd_data(ssd_path: str, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """Load and preprocess SSD data."""
    ssd_df = pd.read_excel(ssd_path)
    
    datetime_cols = ['AlarmStart_rounded_minutes', 'AlarmEnd_rounded_minutes', 
                     'Tag_First_Transition_Start_minutes']
    for col in datetime_cols:
        if col in ssd_df.columns:
            ssd_df[col] = pd.to_datetime(ssd_df[col])
    
    if start_date:
        ssd_df = ssd_df[ssd_df['AlarmStart_rounded_minutes'] >= pd.to_datetime(start_date)]
    if end_date:
        ssd_df = ssd_df[ssd_df['AlarmStart_rounded_minutes'] <= pd.to_datetime(end_date)]
    
    return ssd_df


def load_operating_limits(limits_path: str) -> pd.DataFrame:
    """Load operating limits data."""
    return pd.read_csv(limits_path).set_index('TAG_NAME')


def get_unique_episodes(ssd_df: pd.DataFrame) -> pd.DataFrame:
    """Extract unique alarm episodes from SSD data."""
    episodes = ssd_df.groupby(['AlarmStart_rounded_minutes', 'AlarmEnd_rounded_minutes']).agg({
        'Tag_First_Transition_Start_minutes': 'min'
    }).reset_index()
    
    episodes = episodes.rename(columns={
        'Tag_First_Transition_Start_minutes': 'EarliestTransitionStart'
    })
    episodes = episodes.sort_values('AlarmStart_rounded_minutes').reset_index(drop=True)
    episodes['EpisodeID'] = range(1, len(episodes) + 1)
    
    return episodes


def get_tags_in_episode(ssd_df: pd.DataFrame, alarm_start: pd.Timestamp, 
                        alarm_end: pd.Timestamp) -> list:
    """Get list of tags that transitioned during an episode."""
    mask = (
        (ssd_df['AlarmStart_rounded_minutes'] == alarm_start) & 
        (ssd_df['AlarmEnd_rounded_minutes'] == alarm_end)
    )
    return ssd_df.loc[mask, 'TagName'].unique().tolist()


def create_episode_plot(episode: pd.Series, ts_df: pd.DataFrame, events_df: pd.DataFrame,
                        ssd_df: pd.DataFrame, limits_df: pd.DataFrame,
                        context_minutes: int = 30) -> go.Figure:
    """
    Create comprehensive visualization for a single episode.
    
    Returns Plotly figure object.
    """
    episode_id = episode['EpisodeID']
    transition_start = episode['EarliestTransitionStart']
    alarm_start = episode['AlarmStart_rounded_minutes']
    alarm_end = episode['AlarmEnd_rounded_minutes']
    
    # Define plot time range with context
    plot_start = transition_start - timedelta(minutes=context_minutes)
    plot_end = alarm_end + timedelta(minutes=context_minutes)
    
    # Filter data for plot range
    ts_mask = (ts_df.index >= plot_start) & (ts_df.index <= plot_end)
    ts_plot = ts_df[ts_mask]
    
    events_mask = (
        (events_df['VT_Start'] >= plot_start) & 
        (events_df['VT_Start'] <= plot_end) &
        (events_df['ConditionName'] == 'CHANGE')
    )
    events_plot = events_df[events_mask]
    
    # Get tags involved in this episode
    episode_tags = get_tags_in_episode(ssd_df, alarm_start, alarm_end)
    
    # Create figure with secondary y-axis
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            f'Episode {episode_id}: {alarm_start.strftime("%Y-%m-%d %H:%M")} - {alarm_end.strftime("%H:%M")}',
            'Operator Actions'
        )
    )
    
    # --- Top panel: PV values ---
    
    # Add target tag PV (primary trace)
    target_pv = f'{TARGET_TAG}.PV'
    if target_pv in ts_plot.columns:
        fig.add_trace(
            go.Scatter(
                x=ts_plot.index,
                y=ts_plot[target_pv],
                mode='lines',
                name=f'{TARGET_TAG}.PV (Target)',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
    
    # Add other PV traces (hidden by default)
    colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    color_idx = 0
    
    pv_cols = [col for col in ts_plot.columns if col.endswith('.PV') and col != target_pv]
    for pv_col in pv_cols:
        # Highlight tags that were in the episode
        is_episode_tag = pv_col in episode_tags
        
        fig.add_trace(
            go.Scatter(
                x=ts_plot.index,
                y=ts_plot[pv_col],
                mode='lines',
                name=pv_col + (' (deviated)' if is_episode_tag else ''),
                line=dict(
                    color=colors[color_idx % len(colors)],
                    width=2 if is_episode_tag else 1
                ),
                visible='legendonly' if not is_episode_tag else True
            ),
            row=1, col=1
        )
        color_idx += 1
    
    # Add alarm threshold line
    fig.add_hline(
        y=ALARM_THRESHOLD,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Alarm Limit: {ALARM_THRESHOLD}',
        annotation_position='right',
        row=1, col=1
    )
    
    # Add operating limits for target tag if available
    if target_pv in limits_df.index:
        lower = limits_df.loc[target_pv, 'LOWER_LIMIT']
        upper = limits_df.loc[target_pv, 'UPPER_LIMIT']
        fig.add_hline(y=lower, line_dash='dot', line_color='orange', 
                      annotation_text=f'Lower: {lower:.1f}', row=1, col=1)
        fig.add_hline(y=upper, line_dash='dot', line_color='orange',
                      annotation_text=f'Upper: {upper:.1f}', row=1, col=1)
    
    # Add shaded regions
    # Transition period (light yellow)
    fig.add_vrect(
        x0=transition_start, x1=alarm_start,
        fillcolor='yellow', opacity=0.2,
        layer='below', line_width=0,
        annotation_text='Transition',
        annotation_position='top left',
        row=1, col=1
    )
    
    # Alarm period (light red)
    fig.add_vrect(
        x0=alarm_start, x1=alarm_end,
        fillcolor='red', opacity=0.2,
        layer='below', line_width=0,
        annotation_text='Alarm',
        annotation_position='top left',
        row=1, col=1
    )
    
    # --- Bottom panel: Operator actions timeline ---
    
    # Get unique sources for y-axis positioning
    unique_sources = events_plot['Source'].unique()
    source_to_y = {s: i for i, s in enumerate(unique_sources)}
    
    if len(events_plot) > 0:
        # Determine marker colors based on direction of change
        def get_marker_color(v, p):
            try:
                val = float(v)
                prev = float(p)
                return 'green' if val > prev else 'red'
            except (ValueError, TypeError):
                return 'blue'  # Non-numeric changes
        
        marker_colors = [get_marker_color(v, p) 
                        for v, p in zip(events_plot['Value'].fillna(0), 
                                       events_plot['PrevValue'].fillna(0))]
        
        # Add markers for each action
        fig.add_trace(
            go.Scatter(
                x=events_plot['VT_Start'],
                y=[source_to_y.get(s, 0) for s in events_plot['Source']],
                mode='markers',
                name='Operator Actions',
                marker=dict(
                    size=10,
                    color=marker_colors,
                    symbol='diamond'
                ),
                customdata=events_plot[['Source', 'Description', 'Value', 'PrevValue']].values,
                hovertemplate=(
                    '<b>Time:</b> %{x}<br>' +
                    '<b>Source:</b> %{customdata[0]}<br>' +
                    '<b>Description:</b> %{customdata[1]}<br>' +
                    '<b>Value:</b> %{customdata[2]}<br>' +
                    '<b>PrevValue:</b> %{customdata[3]}<br>' +
                    '<extra></extra>'
                )
            ),
            row=2, col=1
        )
    
    # Add vertical lines for key times
    for time, name, color in [
        (transition_start, 'Transition Start', 'orange'),
        (alarm_start, 'Alarm Start', 'red'),
        (alarm_end, 'Alarm End', 'green')
    ]:
        fig.add_vline(x=time, line_dash='dash', line_color=color, row='all')
    
    # Update layout
    fig.update_layout(
        height=800,
        hovermode='closest',
        showlegend=True,
        legend=dict(
            yanchor='top',
            y=0.99,
            xanchor='left',
            x=1.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        title=dict(
            text=f'Alarm Episode {episode_id}<br><sup>{TARGET_TAG} PVLO Alarm | Duration: {(alarm_end - alarm_start).total_seconds()/60:.0f} min</sup>',
            x=0.5
        )
    )
    
    # Update axes
    fig.update_xaxes(title_text='Time', row=2, col=1)
    fig.update_yaxes(title_text='PV Value', row=1, col=1)
    fig.update_yaxes(
        title_text='Tag',
        ticktext=list(unique_sources)[:20] if len(unique_sources) > 0 else [''],
        tickvals=list(range(min(20, len(unique_sources)))),
        row=2, col=1
    )
    
    return fig


def generate_plots(args):
    """Main function to generate all episode plots."""
    print(f"📊 Episode Visualization Generator")
    print(f"=" * 60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\n📁 Loading data...")
    
    ssd_df = load_ssd_data(args.ssd_file, args.start_date, args.end_date)
    episodes_df = get_unique_episodes(ssd_df)
    print(f"   Episodes: {len(episodes_df)}")
    
    ts_df, events_df, stats = load_all_data(
        ts_path=args.ts_file,
        events_path=args.events_file,
        start_date=args.start_date,
        end_date=args.end_date,
        filter_trips=not args.no_trip_filter,
        verbose=True
    )
    
    limits_df = load_operating_limits(args.operating_limits)
    
    # Determine which episodes to plot
    if args.episode_ids:
        episode_ids = [int(x.strip()) for x in args.episode_ids.split(',')]
        episodes_to_plot = episodes_df[episodes_df['EpisodeID'].isin(episode_ids)]
    elif args.max_episodes:
        episodes_to_plot = episodes_df.head(args.max_episodes)
    else:
        episodes_to_plot = episodes_df
    
    print(f"\n🎨 Generating {len(episodes_to_plot)} visualizations...")
    
    for idx, episode in episodes_to_plot.iterrows():
        episode_id = episode['EpisodeID']
        
        if idx % 10 == 0:
            print(f"   Episode {episode_id}/{len(episodes_to_plot)}...")
        
        try:
            fig = create_episode_plot(
                episode, ts_df, events_df, ssd_df, limits_df,
                context_minutes=args.context_minutes
            )
            
            # Save as HTML
            output_file = output_dir / f'episode_{episode_id:04d}.html'
            fig.write_html(str(output_file), include_plotlyjs='cdn')
            
        except Exception as e:
            print(f"   ⚠️ Error plotting episode {episode_id}: {e}")
    
    # Create index HTML
    index_html = create_index_html(episodes_to_plot, output_dir)
    with open(output_dir / 'index.html', 'w') as f:
        f.write(index_html)
    
    print(f"\n✅ Visualizations saved to {output_dir}/")
    print(f"   Open {output_dir}/index.html to browse episodes")
    
    return len(episodes_to_plot)


def create_index_html(episodes_df: pd.DataFrame, output_dir: Path) -> str:
    """Create index HTML for browsing episodes."""
    rows = []
    for _, ep in episodes_df.iterrows():
        ep_id = ep['EpisodeID']
        start = ep['AlarmStart_rounded_minutes'].strftime('%Y-%m-%d %H:%M')
        end = ep['AlarmEnd_rounded_minutes'].strftime('%H:%M')
        duration = (ep['AlarmEnd_rounded_minutes'] - ep['AlarmStart_rounded_minutes']).total_seconds() / 60
        
        rows.append(f'''
        <tr>
            <td><a href="episode_{ep_id:04d}.html">Episode {ep_id}</a></td>
            <td>{start}</td>
            <td>{end}</td>
            <td>{duration:.0f} min</td>
        </tr>
        ''')
    
    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>Alarm Episode Visualizations</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; max-width: 800px; }}
        th, td {{ padding: 10px; border: 1px solid #ddd; text-align: left; }}
        th {{ background-color: #4CAF50; color: white; }}
        tr:nth-child(even) {{ background-color: #f2f2f2; }}
        a {{ color: #0066cc; text-decoration: none; }}
        a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <h1>🔔 Alarm Episode Visualizations</h1>
    <p>Target: {TARGET_TAG} | Alarm Type: PVLO | Threshold: {ALARM_THRESHOLD}</p>
    <p>Total Episodes: {len(episodes_df)}</p>
    
    <table>
        <tr>
            <th>Episode</th>
            <th>Alarm Start</th>
            <th>Alarm End</th>
            <th>Duration</th>
        </tr>
        {''.join(rows)}
    </table>
</body>
</html>
'''
    return html


if __name__ == '__main__':
    args = parse_args()
    generate_plots(args)
