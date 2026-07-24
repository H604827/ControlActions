"""
Run the validated windowed-difference gain estimator (same method as
EXPERIMENTS/gain_analysis_1071.ipynb Part 1 / Part 8.2) on the NEWLY-available
vendor gain pairs, at 1-MINUTE cadence.
"""
import pandas as pd, numpy as np, re, glob, os
import pyarrow.parquet as pq

DATA    = '/home/h604827/ControlActions/DATA'
RESULTS = '/home/h604827/ControlActions/RESULTS'
NEW_FILE = f'{DATA}/new_rca_pv_op_data/merged_all_historian_tags.parquet'
OLD_DIR  = f'{DATA}/PV-OP_data'
TRIPS    = f'{DATA}/Final_List_Trip_Duration.csv'
DT_MIN   = 1.0                       # these are 1-min files

# (vendor_MV, vendor_CV, our_MV, our_CV, vendor_gain, category, mv_proxy)
PAIRS = [
    ('03PIC1013OP','03PIC1141APV','03PIC_1013.OP','03PI_1141A.PV', -11.2353,'already_validated_30sec',''),
    ('02FI1000FF', '03PIC1141APV','02FI_1000.PV', '03PI_1141A.PV', +74.3774,'already_validated_30sec','FF=PV'),
    ('02FI1000FF', '03TIC1145PV', '02FI_1000.PV', '03TIC_1145.PV', +0.6790,'already_validated_30sec','FF=PV'),
    ('03FIC3435OP','02FI1000PV',  '03FIC_3435.OP','02FI_1000.PV',  +0.0066,'already_validated_30sec','feed as CV; OP from 30sec->1min'),
    ('02FI1000FF', '03TI1081PV',  '02FI_1000.PV', '03TI_1081.PV',  +2.0399,'revived_dead_cv','FF=PV'),
    ('03PIC1013OP','03TI1081PV',  '03PIC_1013.OP','03TI_1081.PV',  -0.0869,'revived_dead_cv',''),
    ('03TIC1092SP','03TI1081PV',  '03TIC_1092.PV','03TI_1081.PV',  -0.4440,'revived_dead_cv','SP->PV proxy'),
    ('02FI1000FF', '03PI3154.PV', '02FI_1000.PV', '03PI_3154.PV',  +0.5043,'new_untested','FF=PV'),
    ('02FI1000FF', '03TI3112PV',  '02FI_1000.PV', '03TI_3112.PV',  +0.6446,'new_untested','FF=PV'),
    ('03FIC3435FF','02PI1220PV',  '03FIC_3435.PV','02PI_1220.PV',  -0.0400,'new_untested','FF=PV'),
    ('03FIC3435FF','03TI3112PV',  '03FIC_3435.PV','03TI_3112.PV',  +0.0609,'new_untested','FF=PV'),
    ('03FIC3435OP','02PI1220PV',  '03FIC_3435.OP','02PI_1220.PV',  -0.0766,'new_untested','OP from 30sec->1min'),
    ('03FIC3435OP','03KM0152IPV', '03FIC_3435.OP','03KM_0152_I.PV',-0.2466,'new_untested','OP from 30sec->1min'),
    ('03TIC1092SP','03LIC3408PV', '03TIC_1092.PV','03LIC_3408.PV', +8.2690,'new_untested','SP->PV proxy'),
    ('02FI1000FF', '03HIC1023AOP','02FI_1000.PV', '03HIC_1023A.OP',+9.4064,'new_untested','FF=PV'),
    ('02FI1000FF', '03LIC3153OP', '02FI_1000.PV', '03LIC_3153.OP', +3.3721,'new_untested','FF=PV'),
    ('02FI1000FF', '03PIC1023OP', '02FI_1000.PV', '03PIC_1023.OP', +6.9361,'new_untested','FF=PV'),
    ('02FI1000FF', '03TI1005PV',  '02FI_1000.PV', '03TI_1005.PV',  +0.2361,'new_untested','FF=PV'),
    ('03PIC1013OP','03HIC1023AOP','03PIC_1013.OP','03HIC_1023A.OP',-1.0989,'new_untested',''),
    ('03TIC1009SP','03HIC1023AOP','03TIC_1009.PV','03HIC_1023A.OP',-2.7995,'new_untested','SP->PV proxy'),
    ('03TIC1009SP','03TI1108PV',  '03TIC_1009.PV','03TI_1108.PV',  +0.9910,'new_untested','SP->PV proxy'),
    ('03TIC1023SP','03PIC1023OP', '03TIC_1023.PV','03PIC_1023.OP', -1.9645,'new_untested','SP->PV proxy'),
    ('03TIC1092SP','03TI1108PV',  '03TIC_1092.PV','03TI_1108.PV',  +0.5496,'new_untested','SP->PV proxy'),
    ('03TIC1092SP','03TIC1009OP', '03TIC_1092.PV','03TIC_1009.OP', -3.4350,'new_untested','SP->PV proxy'),
    ('03TIC1009SP','03TIC1009OP', '03TIC_1009.PV','03TIC_1009.OP', +1.5922,'within_loop_caution','SP->PV proxy, self-loop'),
    ('03TIC1092SP','03TIC1092OP', '03TIC_1092.PV','03TIC_1092.OP', -34.6996,'within_loop_caution','SP->PV proxy, self-loop'),
]

# 03FIC_3435.OP is not in the 1-min per-export files, but it DOES exist in the 30-sec consolidated
# file, so we load it from there and downsample to 1-min (the historian's 1-min series is exactly the
# :00-second subset of the 30-sec data). It is used as the REAL .OP wherever the vendor row is
# 03FIC3435OP -- no .PV substitution. Base de-confounders = the notebook's validated VAL_INPUTS
# (feed + the two clean handles); 03FIC_3435 only enters where it is the MV of interest.
BASE_DECONFOUNDERS = ['02FI_1000.PV', '03PIC_1013.OP', '03TIC_1092.OP']

def canon_instr(col):
    t = col.upper().replace('_', '').replace('.', '').replace(' ', '')
    m = re.match(r'^(\d{1,2})([A-Z]+?)(\d+[A-Z]?\d*)(OP|PV|SP|DV|FF)?$', t)
    if not m:
        return col
    area, letters, num, suf = m.groups()
    return (area, letters[0], num)

# ---- 1. required columns ----------------------------------------------------
need = set(BASE_DECONFOUNDERS)
for _, _, omv, ocv, *_ in PAIRS:
    need.add(omv); need.add(ocv)
need = sorted(need)
print(f'Columns required: {len(need)}')

# ---- 2. load: resolve each column  new-1min -> old-1min -> 30-sec(downsampled) ----
def schema_names(fp):
    return set(pq.ParquetFile(fp).schema_arrow.names)

SEC30_FILE = f'{OLD_DIR}/consolidated_30sec_data/all_tags_30sec.parquet'
new_cols   = schema_names(NEW_FILE)
old_files  = sorted(glob.glob(f'{OLD_DIR}/*.parquet'))
old_schema = {fp: schema_names(fp) for fp in old_files}
sec30_cols = schema_names(SEC30_FILE)

provenance, frames = {}, []

# (a) new merged 1-min file
from_new = [c for c in need if c in new_cols]
dfn = pd.read_parquet(NEW_FILE, columns=['TimeStamp'] + from_new)
dfn['TimeStamp'] = pd.to_datetime(dfn['TimeStamp'], errors='coerce')
dfn = dfn.dropna(subset=['TimeStamp']).drop_duplicates('TimeStamp').set_index('TimeStamp').sort_index()
frames.append(dfn)
provenance.update({c: 'new_merged_1min' for c in from_new})

# (b) old 1-min per-export files
by_src, still = {}, []
for col in [c for c in need if c not in new_cols]:
    src = next((fp for fp in old_files if col in old_schema[fp]), None)
    if src:
        by_src.setdefault(src, []).append(col)
        provenance[col] = f'old_1min:{os.path.basename(src)}'
    else:
        still.append(col)
for src, cols in by_src.items():
    d = pd.read_parquet(src, columns=['TimeStamp'] + cols)
    d['TimeStamp'] = pd.to_datetime(d['TimeStamp'], errors='coerce')
    d = d.dropna(subset=['TimeStamp']).drop_duplicates('TimeStamp').set_index('TimeStamp').sort_index()
    frames.append(d[cols])

# strict 1-min grid from the 1-min sources
ts = pd.concat(frames, axis=1).asfreq('1min')

# (c) 30-sec consolidated file -> downsample to 1-min (:00-second subset == the historian 1-min series)
from_30     = [c for c in still if c in sec30_cols]
missing_all = [c for c in still if c not in sec30_cols]
if from_30:
    d30 = pd.read_parquet(SEC30_FILE, columns=['TimeStamp'] + from_30)
    d30['TimeStamp'] = pd.to_datetime(d30['TimeStamp'], errors='coerce')
    d30 = d30.dropna(subset=['TimeStamp']).drop_duplicates('TimeStamp').set_index('TimeStamp').sort_index()
    for col in from_30:
        ts[col] = d30[col].reindex(ts.index)          # 1-min marks pick the :00-second historian values
        provenance[col] = '30sec->1min'
for col in missing_all:
    print(f'  !! {col} not found in any source -> pairs needing it will be skipped')

have = set(ts.columns)
print('Column provenance:')
for c in need:
    src = provenance.get(c, 'MISSING')
    nn = int(ts[c].notna().sum()) if c in have else 0
    print(f'  {c:18s} <- {src:34s} ({100*nn/len(ts):4.1f}% non-null)')
print(f'\nCombined frame: {ts.shape[0]:,} rows x {ts.shape[1]} cols  ({ts.index.min()} .. {ts.index.max()})')

# ---- 3. trip filter ---------------------------------------------------------
trips = pd.read_csv(TRIPS)
trips['Stop Date']  = pd.to_datetime(trips['Stop Date'],  errors='coerce')
trips['Start Date'] = pd.to_datetime(trips['Start Date'], errors='coerce')
tw = (trips.rename(columns={'Stop Date': 'beg', 'Start Date': 'end'})[['beg', 'end']].dropna())
tw = tw[tw['end'] > tw['beg']]
idx = ts.index.values
mask_trip = np.zeros(len(ts), dtype=bool)
for b, e in tw.itertuples(index=False):
    mask_trip |= (idx >= np.datetime64(b)) & (idx <= np.datetime64(e))
ts.loc[mask_trip, :] = np.nan
print(f'Trip windows applied: {len(tw)} | rows blanked: {mask_trip.sum():,} ({100*mask_trip.mean():.1f}%)')

# ---- 4. estimator (ported, DT_MIN=1) ---------------------------------------
def _diff_table(frame, target, inputs, H_min=60, thin=20, trim=0.001):
    H = max(1, int(round(H_min / DT_MIN)))
    cols = [target] + inputs
    D = pd.DataFrame({c: frame[c].shift(-H) - frame[c] for c in cols})
    D = D.replace([np.inf, -np.inf], np.nan).iloc[::thin].dropna()
    if trim and len(D):
        lo, hi = D[target].quantile(trim), D[target].quantile(1 - trim)
        D = D[(D[target] >= lo) & (D[target] <= hi)]
    return D

def gain_ci(frame, target, inputs, mv, H_min=60, thin=20, trim=0.001, n_boot=200, block=50, seed=0):
    D = _diff_table(frame, target, inputs, H_min, thin, trim)
    n = len(D)
    if n < 50 or mv not in inputs:
        return dict(gain=np.nan, lo=np.nan, hi=np.nan, n=n, r2=np.nan)
    y = D[target].to_numpy()
    X = np.column_stack([D[c].to_numpy() for c in inputs] + [np.ones(n)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r2 = 1.0 - np.var(y - X @ coef) / np.var(y) if np.var(y) > 0 else np.nan
    j = inputs.index(mv)
    rng = np.random.default_rng(seed)
    starts = np.arange(max(1, n - block + 1)); nb = int(np.ceil(n / block)); offs = np.arange(block)
    bs = np.empty(n_boot)
    for k in range(n_boot):
        rows = (rng.choice(starts, size=nb)[:, None] + offs[None, :]).ravel()
        rows = rows[rows < n][:n]
        cb, *_ = np.linalg.lstsq(X[rows], y[rows], rcond=None)
        bs[k] = cb[j]
    lo, hi = np.nanpercentile(bs, [5, 95])
    return dict(gain=float(coef[j]), lo=float(lo), hi=float(hi), n=n, r2=float(r2))

def inputs_for(our_mv, our_cv):
    if our_cv == '02FI_1000.PV':                         # feed as CV -> handles only
        base = ['03PIC_1013.OP', '03TIC_1092.OP']
    else:
        base = BASE_DECONFOUNDERS
    mv_instr = canon_instr(our_mv)
    inp = [our_mv] + [d for d in base if d != our_mv and d != our_cv and canon_instr(d) != mv_instr]
    return list(dict.fromkeys(inp))

# ---- 5. run every pair ------------------------------------------------------
rows = []
for vmv, vcv, omv, ocv, vgain, cat, proxy in PAIRS:
    if omv not in have or ocv not in have:
        rows.append({'category': cat, 'vendor_MV': vmv, 'vendor_CV': vcv, 'our_MV': omv, 'our_CV': ocv,
                     'mv_proxy': proxy, 'vendor_gain': vgain, 'derived_gain': np.nan, 'ci90': 'MISSING COL',
                     'sign_match': None, 'ratio': np.nan, 'R2': np.nan, 'n_win': 0, 'inputs': ''})
        continue
    inputs = [c for c in inputs_for(omv, ocv) if c in have]
    r = gain_ci(ts, ocv, inputs, omv)
    ok = r['gain'] == r['gain']
    rows.append({'category': cat, 'vendor_MV': vmv, 'vendor_CV': vcv, 'our_MV': omv, 'our_CV': ocv,
                 'mv_proxy': proxy, 'vendor_gain': vgain,
                 'derived_gain': round(r['gain'], 4) if ok else np.nan,
                 'ci90': f"[{r['lo']:+.3f}, {r['hi']:+.3f}]" if ok else 'n/a',
                 'sign_match': (bool(np.sign(r['gain']) == np.sign(vgain)) if ok else None),
                 'ratio': round(r['gain'] / vgain, 2) if (ok and vgain != 0) else np.nan,
                 'R2': round(r['r2'], 3) if r['r2'] == r['r2'] else np.nan,
                 'n_win': r['n'], 'inputs': ' + '.join(inputs)})

res = pd.DataFrame(rows)
cat_order = {'already_validated_30sec': 0, 'revived_dead_cv': 1, 'new_untested': 2, 'within_loop_caution': 3}
res['_o'] = res['category'].map(cat_order)
res = res.sort_values(['_o', 'vendor_CV', 'vendor_MV']).drop(columns='_o').reset_index(drop=True)

pd.set_option('display.max_rows', 300); pd.set_option('display.width', 240)
print('\n\n============ GAIN ESTIMATOR ON NEW 1-MIN PAIRS (H=60, thin=20, trip-filtered) ============\n')
print(res[['category', 'our_MV', 'our_CV', 'mv_proxy', 'vendor_gain', 'derived_gain',
           'ci90', 'sign_match', 'ratio', 'R2', 'n_win']].to_string(index=False))

info = res[(res.category != 'within_loop_caution') & res.sign_match.notna()]
print(f"\nSign-match (excl. within-loop & missing): {int(info['sign_match'].sum())}/{len(info)}")
good = info[(info['R2'] >= 0.10)]
print(f"Sign-match among R2>=0.10 anchors        : {int(good['sign_match'].sum())}/{len(good)}")

OUT = f'{RESULTS}/gain_1min_new_pairs_results.xlsx'
readme = pd.DataFrame({'How to read this file': [
    "Derived process gains for the newly-available vendor gain pairs, at 1-MINUTE cadence, using the SAME",
    "de-confounded windowed-difference estimator validated in EXPERIMENTS/gain_analysis_1071.ipynb.",
    "Data: DATA/new_rca_pv_op_data/merged_all_historian_tags.parquet (+ old 1-min PV-OP_data files for",
    "  03HIC_1023A.OP, + the 30-sec consolidated file downsampled to 1-min for 03FIC_3435.OP). Strict",
    "  1-min grid, trip windows removed. H=60 min, thin=20, trim 0.1%.",
    "derived_gain = coefficient on our_MV in the joint regression d60(CV)=sum_i g_i*d60(MV_i)+c",
    "  (jointly fitting all inputs gives the partial derivative dCV/dMV = the de-confounded gain).",
    "ci90 = 90% moving-block bootstrap CI. R2 = whole-model fit (a trust proxy for that CV).",
    "sign_match = derived gain has SAME SIGN as vendor (direction = the reliable part). ratio = derived/vendor.",
    "Magnitude gaps ~2-5x are structural (closed-loop rejection, omitted MVs, throughput co-movement, units).",
    "",
    "mv_proxy:",
    "  FF=PV        -> vendor feed-forward row; the FF signal IS the measured value, so the regressor is",
    "                  the loop's .PV. This is the correct native mapping, NOT an OP<->PV substitution.",
    "  SP->PV proxy -> vendor SETPOINT row; no .SP available, regressor = loop .PV (PV\u2248SP at steady state).",
    "                  Physically-correct proxy for cascade loops (event-reconstructed SP is unreliable).",
    "                  Read these as INDICATIVE (direction more than magnitude).",
    "  OP from 30sec->1min -> 03FIC_3435.OP is absent from the 1-min per-export files but present in the",
    "                  30-sec consolidated file; loaded from there and downsampled to 1-min (:00-second",
    "                  subset = the historian 1-min series). REAL .OP data, no substitution.",
    "  feed as CV   -> 3435.OP regressed with the feed as the CV (handles-only inputs).",
    "  self-loop    -> within_loop_caution: MV and CV are the SAME instrument (e.g. 03TIC_1092 SP vs its own",
    "                  OP) = controller INVERSE coupling, NOT a process gain; the large vendor numbers",
    "                  (e.g. -34.7) are not reproducible by design.",
    "",
    "category: already_validated_30sec (1-min consistency check) | revived_dead_cv (03TI_1081, dead in 30-sec)",
    "  | new_untested (first numbers here) | within_loop_caution (see self-loop).",
]})
with pd.ExcelWriter(OUT) as xl:
    res.to_excel(xl, sheet_name='gain_estimates_1min', index=False)
    readme.to_excel(xl, sheet_name='How_to_read', index=False)
print(f'\nSaved -> {OUT}')
