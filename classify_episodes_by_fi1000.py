import pandas as pd
import os
import shutil
from pathlib import Path

SOURCE_DIR = Path('/home/h604827/ControlActions/RESULTS/03LIC_1071_episodes/episode_visualizations')
OUTPUT_DIR = Path('/home/h604827/ControlActions/RESULTS/03LIC_1071_episodes_by_FI1000')

FI1000_COL = '02FI_1000.PV'

# Ranges: (lower, upper) — both inclusive
RANGES = [
    (8.0, 8.25),
    (8.25, 8.5),
    (8.5, 8.75),
    (8.75, 9.0),
]

# Create output directories
for lo, hi in RANGES:
    folder_name = f'FI1000_{lo:.2f}_{hi:.2f}'
    (OUTPUT_DIR / folder_name).mkdir(parents=True, exist_ok=True)

# Process each episode
episode_dirs = sorted([d for d in SOURCE_DIR.iterdir() if d.is_dir() and d.name.startswith('episode_')])
print(f'Total episode folders: {len(episode_dirs)}')

stats = {r: [] for r in RANGES}
no_data = []
no_column = []

for ep_dir in episode_dirs:
    ep_name = ep_dir.name
    # Find pv_data CSV
    pv_files = list(ep_dir.glob('*_pv_data.csv'))
    if not pv_files:
        no_data.append(ep_name)
        continue
    
    pv_file = pv_files[0]
    df = pd.read_csv(pv_file)
    
    if FI1000_COL not in df.columns:
        no_column.append(ep_name)
        continue
    
    fi_values = df[FI1000_COL].dropna()
    
    if fi_values.empty:
        no_data.append(ep_name)
        continue
    
    fi_min = fi_values.min()
    fi_max = fi_values.max()
    
    # Check each range: ALL values must be within [lo, hi]
    for (lo, hi) in RANGES:
        if fi_min >= lo and fi_max <= hi:
            folder_name = f'FI1000_{lo:.2f}_{hi:.2f}'
            dest = OUTPUT_DIR / folder_name / ep_name
            # Create symlink to original episode folder
            if not dest.exists():
                os.symlink(ep_dir, dest)
            stats[(lo, hi)].append(ep_name)

print(f'\nEpisodes without PV data: {len(no_data)}')
print(f'Episodes without {FI1000_COL} column: {len(no_column)}')
print(f'\n{"="*60}')
print(f'Classification results:')
print(f'{"="*60}')
for (lo, hi) in RANGES:
    folder_name = f'FI1000_{lo:.2f}_{hi:.2f}'
    n = len(stats[(lo, hi)])
    print(f'  {folder_name}: {n} episodes')

# Episodes not in any range
all_classified = set()
for eps in stats.values():
    all_classified.update(eps)
unclassified = len(episode_dirs) - len(no_data) - len(no_column) - len(all_classified)
print(f'\n  Episodes in no range (FI1000 spans multiple ranges): {len(episode_dirs) - len(no_data) - len(no_column) - len(all_classified)}')
print(f'\nDone! Output: {OUTPUT_DIR}')
