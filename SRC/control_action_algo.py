"""
Control Action Algorithm for 03LIC_1071 PVLO Alarm Prevention & Resolution

First-principles, per-cause supervisory recommender for the OTS (Operator Training
Simulator). The OTS is a high-fidelity dynamic + DCS (PID) model; it does NOT include
the APC/MPC layer, so this algorithm acts directly on the regulatory loops.

OTS faults (all injected with controllers in AUTO):
  1-3. Feed Gas Disturbance (FI1000 high, ~500/480/460 T/day) -> 03LIC_1071 level falls.
  4.   Compressor-speed disturbance (03PIC_1013.OP up -> suction pressure falls) -> level falls.

Engine (three independent pieces):
  - DIRECTION  : fixed by the 3E107 mass balance (physics), not data fitting.
                 Feed cause     -> INCREASE 03LIC_1071 (open inflow to raise level).
                 Pressure cause -> DECREASE 03PIC_1013 OP (lower compressor speed ->
                 raise suction pressure -> suppress propane flashing -> level recovers).
  - LEVER      : selected from the RCA cause tag(s) (cause -> lever map; 1071-OP fallback).
  - MAGNITUDE  : severity-scaled proportional law,
                 magnitude = severity_factor x base_step x aggressiveness,
                 where severity = depth below 28.75 + rate of fall.

Supply cascade: 1071 draws liquid from tank 1016. Opening 1071 inflow draws 1016 down,
so a guard raises 03LIC_1016 OP when 1016 level nears its low limit (or 1071 OP saturates).

Trigger modes:
- PREVENTION: Triggered before alarm (PV still above 28.75 but falling)
- RESOLUTION: Triggered after alarm fires (PV already below 28.75)
"""

import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ==============================================================================
# CONSTANTS — derived from 3.5 years of historical data
# ==============================================================================

ALARM_THRESHOLD = 28.75  # PVLO alarm limit
OPERATING_LOW = 35.25    # Lower operating limit (from operating_limits.csv)
OPERATING_HIGH = 42.41   # Upper operating limit
SETPOINT = 38.98         # Mid-range target (median of normal operation)

# Rate of change thresholds (PV units per minute, based on historical distributions)
# Normal operation: std = 0.57/min, 95th percentile = ±0.88/min
# Pre-alarm: mean = -0.095/min, Q10 = -1.5/min
ROC_NOISE_FLOOR = 0.10     # Below this, no action (normal fluctuation)
ROC_MILD = 0.30            # Mild concern — monitor closely
ROC_MODERATE = 0.60        # Moderate — take gentle action
ROC_SEVERE = 1.50          # Severe — aggressive action
ROC_CRITICAL = 3.00        # Critical — maximum response

# Response lags (minutes) — from lag correlation analysis
RESPONSE_LAG_LIC_1071 = 12   # Self-controller: OP change → PV response
RESPONSE_LAG_PIC_1013 = 10   # Pressure path
RESPONSE_LAG_LOAD_REDUCTION = 15  # HIC/load tags (slower, indirect path)
RESPONSE_LAG_RETENTION = 8   # LIC_3178, LIC_3153 (direct liquid retention)

# OP range constraints (from historical data)
OP_LIMITS = {
    '03LIC_1071': {'min': 0.0, 'max': 85.0, 'normal': 51.0},
    '03PIC_1013': {'min': 0.0, 'max': 97.0, 'normal': 83.0},
    '03HIC_3100': {'min': 0.0, 'max': 100.0, 'normal': None},
    '03HIC_1151': {'min': 0.0, 'max': 100.0, 'normal': None},
    '03HIC_1141': {'min': 0.0, 'max': 100.0, 'normal': None},
    '02HIC_1087': {'min': 0.0, 'max': 100.0, 'normal': None},
    '02HIC_1050': {'min': 0.0, 'max': 100.0, 'normal': None},
    '03LIC_3178': {'min': 0.0, 'max': 105.0, 'normal': 40.0},
    '03LIC_3153': {'min': 0.0, 'max': 100.0, 'normal': None},
    '03FIC_3435': {'min': 0.0, 'max': 100.0, 'normal': None},
    '03FIC_1085': {'min': 0.0, 'max': 105.0, 'normal': 38.0},
    # 1016 supply tank — feeds liquid to 1071 (cascade guard target)
    '03LIC_1016': {'min': 0.0, 'max': 100.0, 'normal': 38.0},
}

# --- 1016 supply-tank guard (03LIC_1016 supplies liquid to 03LIC_1071) ---
# Values approximate; confirm against live OTS limits.
LEVEL_LOW_1016 = 38.0      # 03LIC_1016.PV operating-low limit
LEVEL_MARGIN_1016 = 1.0    # start guarding this far above the low limit
OP_SAT_MARGIN = 3.0        # how close to OP max counts as 'saturated'
SP_MAX_STEP = 3.0          # cap on a single 03LIC_1071 SP step (level %)
BASE_STEP = 2.0            # median operator OP step (physically sensible, not fitted)

# --- Cause -> lever mapping (RCA-driven) ---
# RCA hands us the cause tag(s); each maps to one primary first-principles lever.
LEVERS = {
    'feed': {
        'name': 'feed',
        'tag': '03LIC_1071',
        'raise_direction': 'increase',   # increase 1071 to raise level
        'base_step': BASE_STEP,
        'response_lag': RESPONSE_LAG_LIC_1071,
        'allow_sp': True,                # may use SP (AUTO) or OP (MANUAL)
    },
    'pressure': {
        'name': 'pressure',
        'tag': '03PIC_1013',
        'raise_direction': 'decrease',   # decrease 1013 OP to raise suction pressure
        'base_step': BASE_STEP,
        'response_lag': RESPONSE_LAG_PIC_1013,
        'allow_sp': False,               # operators never move 1013 SP — OP only
    },
}


def _normalize_tag(tag) -> str:
    """Uppercase, drop .PV/.OP/.SP suffix and underscores for tolerant matching."""
    t = str(tag).upper().strip()
    for suf in ('.PV', '.OP', '.SP'):
        if t.endswith(suf):
            t = t[:-3]
    return t.replace(' ', '').replace('_', '')


def map_cause_to_lever(cause_tag) -> Optional[str]:
    """Map an RCA cause tag to a lever name ('feed'/'pressure'), or None if unknown."""
    n = _normalize_tag(cause_tag)
    if 'PIC1013' in n:
        return 'pressure'
    if 'FI1000' in n:
        return 'feed'
    return None


# ==============================================================================
# DATA TYPES
# ==============================================================================

class Severity(Enum):
    NONE = 0
    MILD = 1
    MODERATE = 2
    SEVERE = 3
    CRITICAL = 4


class ActionType(Enum):
    MODE_CHANGE = "MODE"
    OP_CHANGE = "OP"
    SP_CHANGE = "SP"


@dataclass
class PlantState:
    """Snapshot of current plant state at a single timestamp."""
    timestamp: object  # pd.Timestamp
    pv_1071: float     # Current level PV (%)
    op_1071: float     # Current level controller OP (%)
    pv_pic_1013: Optional[float] = None  # Current pressure (bar); optional
    op_pic_1013: Optional[float] = None  # Current pressure controller OP (%); optional
    fi_1000: Optional[float] = None      # Current feed rate; optional (unused by engine)

    # Rate of change (computed over rolling window)
    roc_1071: float = 0.0     # d(PV)/dt for level (per minute)
    roc_pic_1013: float = 0.0  # d(PV)/dt for pressure

    # Optional: other tags for richer context
    pv_lic_3178: Optional[float] = None
    op_lic_3178: Optional[float] = None
    pv_fic_3435: Optional[float] = None

    # Supply tank 1016 (feeds 1071) — for the cascade guard
    pv_1016: Optional[float] = None
    op_1016: Optional[float] = None

    # 1071 setpoint and loop mode — for the OP-vs-SP decision
    sp_1071: Optional[float] = None
    mode_1071: Optional[str] = None   # 'AUTO' / 'MAN' (None -> assume AUTO)


@dataclass
class ControlAction:
    """A single control action to execute."""
    tag: str                    # Tag to act on (e.g., '03LIC_1071')
    action_type: ActionType     # OP_CHANGE or SP_CHANGE
    value: float               # Step SIZE to apply (caller adds to current OP/SP)
    direction: str             # 'increase' or 'decrease'
    magnitude: float           # Absolute change in OP/SP units
    reason: str                # Human-readable explanation
    priority: int              # Execution order (1 = first)
    response_lag_minutes: int  # Expected time before effect visible
    mode_to_manual: bool = False        # Switch loop to MANUAL before applying (OP moves)
    target_value: Optional[float] = None  # Concrete target OP/SP when known (clamped)
    
    @property
    def step(self) -> float:
        """Signed step change."""
        return self.magnitude if self.direction == 'increase' else -self.magnitude


@dataclass
class AlgorithmOutput:
    """Full output of one algorithm scan."""
    timestamp: object
    severity: Severity
    time_to_alarm_minutes: float  # Estimated time until alarm breach (inf if rising)
    actions: list = field(default_factory=list)
    diagnosis: str = ""  # What's happening


# ==============================================================================
# CORE ALGORITHM
# ==============================================================================

class ControlActionAlgorithm:
    """
    Supervisory control algorithm for 03LIC_1071 PVLO alarm prevention/resolution.
    
    Design principles:
    1. Rate-of-fall is the primary severity signal
    2. Severity determines which tiers of action to engage
    3. Action magnitude scales linearly with severity (no derivative kick, no integral windup)
    4. After acting, wait for response_lag before re-evaluating that tag
    5. Conservative: would rather under-act and re-evaluate than over-correct
    
    The algorithm is stateless scan-to-scan (no internal memory of past actions).
    The caller is responsible for not re-calling within the response lag window.
    """

    def __init__(self,
                 levers: Optional[dict] = None,
                 aggressiveness: float = 1.0):
        """
        Parameters
        ----------
        levers : dict, optional
            Override the default lever configuration (keyed by lever name).
            Defaults to LEVERS: feed -> 03LIC_1071, pressure -> 03PIC_1013.
        aggressiveness : float
            Multiplier on action magnitude (1.0 = nominal, 0.5 = conservative,
            2.0 = aggressive). The single OTS-tuning knob.
        """
        self.aggressiveness = aggressiveness
        self.levers = levers if levers is not None else LEVERS

    def all_lever_tags(self) -> list:
        """Tags the algorithm may ever act on (levers + the 1016 supply guard)."""
        tags = [cfg['tag'] for cfg in self.levers.values()]
        if '03LIC_1016' not in tags:
            tags.append('03LIC_1016')
        return tags

    def select_levers(self, rca_cause_tags=None) -> list:
        """
        Map RCA cause tag(s) to lever config(s).

        If no cause tags are given, fall back to the feed lever — raising the
        level via 03LIC_1071 is the universal remedy for a low level.
        Any cause tag we don't recognise also falls back to the feed lever.
        """
        if not rca_cause_tags:
            return [self.levers['feed']]
        selected, seen = [], set()
        for tag in rca_cause_tags:
            name = map_cause_to_lever(tag) or 'feed'
            if name in self.levers and name not in seen:
                selected.append(self.levers[name])
                seen.add(name)
        return selected or [self.levers['feed']]

    def _magnitude(self, severity_factor: float, base_step: float = BASE_STEP) -> float:
        """severity_factor x base_step x aggressiveness; floored at 0.5, rounded to 0.5."""
        mag = severity_factor * base_step * self.aggressiveness
        if mag < 0.5:
            return 0.0
        return round(mag * 2) / 2

    def _clamp(self, tag: str, value: float) -> float:
        lim = OP_LIMITS.get(tag, {'min': 0.0, 'max': 100.0})
        return max(lim['min'], min(lim['max'], value))

    def assess_severity(self, state: PlantState) -> tuple:
        """
        Determine severity level from current plant state.
        
        Uses two signals:
        1. Rate of change (how fast is level falling?)
        2. Proximity to alarm (how close to 28.75?)
        
        Returns (Severity, time_to_alarm_minutes, diagnosis)
        """
        pv = state.pv_1071
        roc = state.roc_1071  # negative = falling
        
        # --- Time to alarm estimation ---
        margin = pv - ALARM_THRESHOLD  # positive = above alarm
        
        if roc >= 0:
            # Level is rising or stable — no alarm threat
            time_to_alarm = float('inf')
        elif margin <= 0:
            # Already in alarm
            time_to_alarm = 0.0
        else:
            # Falling towards alarm
            time_to_alarm = margin / abs(roc)  # minutes until threshold breach
        
        # --- Severity classification ---
        # Combine rate magnitude with proximity
        rate_magnitude = abs(min(roc, 0))  # only care about falling
        
        if margin <= 0:
            # Already in alarm — RESOLUTION mode.
            # Keep pushing until PV climbs back above the threshold. Do NOT
            # de-escalate just because the fall has stopped: a level sitting
            # below 28.75 is still an active alarm that must be resolved.
            # Severity is driven by how deep below the limit we are AND the rate.
            depth = ALARM_THRESHOLD - pv  # >= 0, how far below the alarm limit

            if depth > 8.0 or rate_magnitude > ROC_SEVERE:
                severity = Severity.CRITICAL
            elif depth > 3.0 or rate_magnitude > ROC_MODERATE:
                severity = Severity.SEVERE
            else:
                # Shallow and not falling fast — still in alarm, keep acting
                severity = Severity.MODERATE

            if rate_magnitude <= ROC_NOISE_FLOOR:
                trend = "flat"
            elif roc < 0:
                trend = f"still falling at {roc:.3f}/min"
            else:
                trend = f"recovering at {roc:.3f}/min"
            diagnosis = (f"IN ALARM (resolution). PV={pv:.1f}, "
                         f"{depth:.1f} below limit, {trend}")
        
        elif time_to_alarm < 5:
            # About to alarm (< 5 min)
            severity = Severity.CRITICAL if rate_magnitude > ROC_MODERATE else Severity.SEVERE
            diagnosis = f"IMMINENT alarm in ~{time_to_alarm:.0f}min. PV={pv:.1f}, roc={roc:.3f}/min"
        
        elif time_to_alarm < 15:
            # Approaching alarm (5-15 min)
            severity = Severity.SEVERE if rate_magnitude > ROC_MODERATE else Severity.MODERATE
            diagnosis = f"Approaching alarm in ~{time_to_alarm:.0f}min. PV={pv:.1f}, roc={roc:.3f}/min"
        
        elif time_to_alarm < 30:
            # Early warning (15-30 min)
            if rate_magnitude > ROC_MILD:
                severity = Severity.MILD
                diagnosis = f"Early warning. PV={pv:.1f}, roc={roc:.3f}/min, alarm in ~{time_to_alarm:.0f}min"
            else:
                severity = Severity.NONE
                diagnosis = f"Slow drift. PV={pv:.1f}, roc={roc:.3f}/min"
        
        else:
            severity = Severity.NONE
            diagnosis = f"Normal. PV={pv:.1f}, roc={roc:.3f}/min"
        
        return severity, time_to_alarm, diagnosis

    def compute_severity_factor(self, severity: Severity, time_to_alarm: float) -> float:
        """
        Convert severity into a 0→1 scaling factor for action magnitude.
        
        This is the core "how aggressive should we be?" calculation.
        Linear ramp within each severity band, continuous across bands.
        """
        factors = {
            Severity.NONE: 0.0,
            Severity.MILD: 0.25,
            Severity.MODERATE: 0.50,
            Severity.SEVERE: 0.75,
            Severity.CRITICAL: 1.0,
        }
        base = factors[severity]
        
        # Already in alarm at CRITICAL — full power
        if severity == Severity.CRITICAL and time_to_alarm <= 0:
            return 1.0
        # Urgency boost in the final minutes before the alarm (prevention only)
        if 0 < time_to_alarm < 5:
            return base + (1.0 - base) * (1 - time_to_alarm / 5)
        
        return base

    def _choose_1071_variable(self, severity: Severity, state: PlantState) -> str:
        """
        Decide whether to move 03LIC_1071 via OP (MODE->MANUAL) or SP (in AUTO).

        - In alarm, or SEVERE/CRITICAL (deep/fast): MODE->MANUAL + OP for a fast,
          direct level boost.
        - Mild/moderate prevention with the loop in AUTO and a known SP: nudge the
          SP up gently and let the PID do the work.
        - If the SP is unknown, default to OP.
        """
        in_alarm = state.pv_1071 < ALARM_THRESHOLD
        if in_alarm or severity.value >= Severity.SEVERE.value:
            return 'OP'
        if state.sp_1071 is not None and (state.mode_1071 in (None, 'AUTO')):
            return 'SP'
        return 'OP'

    def _feed_actions(self, state: PlantState, severity: Severity,
                      severity_factor: float) -> list:
        """Feed lever: raise level via 03LIC_1071 (OP or SP) + 1016 supply guard."""
        actions = []
        mag = self._magnitude(severity_factor)
        if mag > 0:
            variable = self._choose_1071_variable(severity, state)
            if variable == 'OP':
                cur = state.op_1071
                target = self._clamp('03LIC_1071', cur + mag) if cur is not None else None
                actions.append(ControlAction(
                    tag='03LIC_1071', action_type=ActionType.OP_CHANGE,
                    value=mag, direction='increase', magnitude=mag,
                    reason='Feed disturbance: open 03LIC_1071 inflow to raise level',
                    priority=0, response_lag_minutes=RESPONSE_LAG_LIC_1071,
                    mode_to_manual=(state.mode_1071 != 'MAN'), target_value=target))
            else:
                sp_mag = min(mag, SP_MAX_STEP)
                cur = state.sp_1071
                target = (cur + sp_mag) if cur is not None else None
                actions.append(ControlAction(
                    tag='03LIC_1071', action_type=ActionType.SP_CHANGE,
                    value=sp_mag, direction='increase', magnitude=sp_mag,
                    reason='Feed disturbance: raise 03LIC_1071 level setpoint (gentle, AUTO)',
                    priority=0, response_lag_minutes=RESPONSE_LAG_LIC_1071,
                    mode_to_manual=False, target_value=target))
        guard = self._supply_guard_1016(state, severity_factor)
        if guard is not None:
            actions.append(guard)
        return actions

    def _pressure_actions(self, state: PlantState, severity_factor: float) -> list:
        """Pressure lever: decrease 03PIC_1013 OP to raise suction pressure."""
        mag = self._magnitude(severity_factor)
        if mag <= 0:
            return []
        cur = state.op_pic_1013
        target = self._clamp('03PIC_1013', cur - mag) if cur is not None else None
        return [ControlAction(
            tag='03PIC_1013', action_type=ActionType.OP_CHANGE,
            value=mag, direction='decrease', magnitude=mag,
            reason=('Compressor-speed disturbance: lower 03PIC_1013 OP -> reduce speed '
                    '-> raise suction pressure -> suppress propane flashing -> level recovers'),
            priority=0, response_lag_minutes=RESPONSE_LAG_PIC_1013,
            mode_to_manual=True, target_value=target)]

    def _supply_guard_1016(self, state: PlantState, severity_factor: float):
        """
        Cascade guard: 1071 draws liquid from tank 1016. When 1016 nears its low
        limit (or 1071 OP is saturated and can't pull more), raise 03LIC_1016 OP
        to refill the supply tank so 1071 keeps a liquid source.
        """
        if state.pv_1016 is None:
            return None
        floor = LEVEL_LOW_1016 + LEVEL_MARGIN_1016
        low_1016 = state.pv_1016 < floor
        op_1071_sat = (state.op_1071 is not None and
                       state.op_1071 >= OP_LIMITS['03LIC_1071']['max'] - OP_SAT_MARGIN)
        if not (low_1016 or op_1071_sat):
            return None
        mag = self._magnitude(severity_factor)
        if mag <= 0:
            return None
        cur = state.op_1016
        target = self._clamp('03LIC_1016', cur + mag) if cur is not None else None
        why = []
        if low_1016:
            why.append(f'1016 level {state.pv_1016:.1f} near low limit {LEVEL_LOW_1016:.1f}')
        if op_1071_sat:
            why.append('1071 OP near saturation')
        return ControlAction(
            tag='03LIC_1016', action_type=ActionType.OP_CHANGE,
            value=mag, direction='increase', magnitude=mag,
            reason='Supply guard: refill 03LIC_1016 (feeds 1071) - ' + '; '.join(why),
            priority=0, response_lag_minutes=RESPONSE_LAG_LIC_1071,
            mode_to_manual=True, target_value=target)


    def compute_actions(self, state: PlantState, rca_cause_tags=None) -> AlgorithmOutput:
        """
        Main entry point. Given the current state and the RCA cause tag(s),
        produce severity-scaled control actions on the cause-selected lever(s).

        Parameters
        ----------
        state : PlantState
            Current snapshot of relevant tag values and rates.
        rca_cause_tags : list, optional
            Cause tag(s) from the upstream RCA module. If None, defaults to the
            feed lever (03LIC_1071) — raising the level is the universal remedy.

        Returns
        -------
        AlgorithmOutput with severity, time_to_alarm, and a list of ControlActions.
        """
        severity, time_to_alarm, diagnosis = self.assess_severity(state)

        output = AlgorithmOutput(
            timestamp=state.timestamp,
            severity=severity,
            time_to_alarm_minutes=time_to_alarm,
            diagnosis=diagnosis,
            actions=[],
        )

        if severity == Severity.NONE:
            return output

        sf = self.compute_severity_factor(severity, time_to_alarm)
        levers = self.select_levers(rca_cause_tags)

        all_actions = []
        for lever in levers:
            if lever['name'] == 'feed':
                all_actions.extend(self._feed_actions(state, severity, sf))
            elif lever['name'] == 'pressure':
                all_actions.extend(self._pressure_actions(state, sf))

        for i, action in enumerate(all_actions, start=1):
            action.priority = i
        output.actions = all_actions

        return output


# ==============================================================================
# RATE OF CHANGE ESTIMATOR
# ==============================================================================

def compute_rate_of_change(pv_history: np.ndarray, 
                           window: int = 5,
                           method: str = 'ema') -> float:
    """
    Compute rate of change from a window of PV values (1-min resolution).
    
    Parameters
    ----------
    pv_history : array-like
        Last N minutes of PV values (most recent last).
        Must have at least `window` values.
    window : int
        Number of minutes for rate calculation.
    method : str
        'simple' — (pv[-1] - pv[-window]) / window
        'ema' — exponentially-weighted slope (less noisy)
        'regression' — linear regression slope over window (most robust)
    
    Returns
    -------
    float : rate of change in PV units per minute (negative = falling)
    """
    if len(pv_history) < window:
        return 0.0
    
    recent = np.array(pv_history[-window:], dtype=float)
    
    # Remove NaN
    valid = ~np.isnan(recent)
    if valid.sum() < 3:
        return 0.0
    
    if method == 'simple':
        return (recent[-1] - recent[0]) / (window - 1)
    
    elif method == 'ema':
        # EMA-weighted difference: emphasizes recent values
        weights = np.exp(np.linspace(-1, 0, window))
        weights = weights / weights.sum()
        weighted_diff = np.diff(recent)
        if len(weighted_diff) == 0:
            return 0.0
        w = weights[1:]  # weights for differences
        w = w / w.sum()
        return float(np.sum(weighted_diff * w))
    
    elif method == 'regression':
        # Linear regression: most robust to noise
        x = np.arange(window, dtype=float)
        y = recent
        # Only use valid points
        x_valid = x[valid]
        y_valid = y[valid]
        if len(x_valid) < 3:
            return 0.0
        # Slope via least squares
        x_mean = x_valid.mean()
        y_mean = y_valid.mean()
        slope = np.sum((x_valid - x_mean) * (y_valid - y_mean)) / np.sum((x_valid - x_mean)**2)
        return float(slope)  # units per minute (since x is in minutes)
    
    else:
        raise ValueError(f"Unknown method: {method}")


# ==============================================================================
# CONVENIENCE: Build PlantState from a DataFrame row/slice
# ==============================================================================

def build_state_from_data(pv_data_slice, timestamp, roc_window=5):
    """
    Construct a PlantState from a slice of the PV/OP DataFrame.
    
    Parameters
    ----------
    pv_data_slice : pd.DataFrame
        Slice of PV/OP data (indexed by TimeStamp) covering at least 
        roc_window minutes up to and including the target timestamp.
    timestamp : pd.Timestamp
        The current scan time.
    roc_window : int
        Minutes of history for rate of change calculation.
    
    Returns
    -------
    PlantState
    """
    # Current values
    try:
        row = pv_data_slice.loc[timestamp]
    except KeyError:
        # Find nearest timestamp
        idx = pv_data_slice.index.get_indexer([timestamp], method='nearest')[0]
        row = pv_data_slice.iloc[idx]
    
    # Safe scalar getter: returns None if the column is missing or the value is NaN
    def _get(col):
        if col not in pv_data_slice.columns:
            return None
        v = row[col]
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return None if f != f else f

    pv_1071 = _get('03LIC_1071.PV')
    op_1071 = _get('03LIC_1071.OP')
    if pv_1071 is None or op_1071 is None:
        raise KeyError("build_state_from_data requires 03LIC_1071.PV and 03LIC_1071.OP")

    # Rate of change over window
    window_start = timestamp - np.timedelta64(roc_window, 'm')
    pv_window = pv_data_slice.loc[window_start:timestamp, '03LIC_1071.PV'].values
    roc_1071 = compute_rate_of_change(pv_window, window=roc_window, method='regression')

    roc_pic_1013 = 0.0
    if '03PIC_1013.PV' in pv_data_slice.columns:
        pic_window = pv_data_slice.loc[window_start:timestamp, '03PIC_1013.PV'].values
        roc_pic_1013 = compute_rate_of_change(pic_window, window=roc_window, method='regression')

    state = PlantState(
        timestamp=timestamp,
        pv_1071=pv_1071,
        op_1071=op_1071,
        pv_pic_1013=_get('03PIC_1013.PV'),
        op_pic_1013=_get('03PIC_1013.OP'),
        fi_1000=_get('02FI_1000.PV'),
        roc_1071=roc_1071,
        roc_pic_1013=roc_pic_1013,
    )
    
    # Optional tags
    if '03LIC_3178.PV' in pv_data_slice.columns:
        state.pv_lic_3178 = float(row['03LIC_3178.PV'])
    if '03LIC_3178.OP' in pv_data_slice.columns:
        state.op_lic_3178 = float(row['03LIC_3178.OP'])
    if '03FIC_3435.PV' in pv_data_slice.columns:
        state.pv_fic_3435 = float(row['03FIC_3435.PV'])

    # Supply tank 1016, plus 1071 setpoint (all optional in the data)
    for col, attr in (('03LIC_1016.PV', 'pv_1016'),
                      ('03LIC_1016.OP', 'op_1016'),
                      ('03LIC_1071.SP', 'sp_1071')):
        if col in pv_data_slice.columns:
            v = row[col]
            if v == v:  # skip NaN (NaN != NaN)
                setattr(state, attr, float(v))
    if '03LIC_1071.MODE' in pv_data_slice.columns:
        m = row['03LIC_1071.MODE']
        if isinstance(m, str):
            state.mode_1071 = m

    return state


# ==============================================================================
# SNAPSHOT ENTRY POINT — run the algorithm on a time-series snapshot
# ==============================================================================

def suggest_actions_from_snapshot(df, algo=None, roc_window=5,
                                  timestamp_col='TimeStamp', rca_cause_tags=None):
    """
    Run the control-action algorithm on a time-series snapshot that ends at (or
    just before) the moment of interest — e.g. data ending just before the
    03LIC_1071 PVLO alarm limit is hit.

    This is the PREVENTION entry point: give it the recent history of all tags
    and it returns the control actions the algorithm would recommend right now.

    Parameters
    ----------
    df : pd.DataFrame
        Time series of all tags. Must contain at least:
          - '03LIC_1071.PV', '03LIC_1071.OP'
          - '03PIC_1013.PV', '03PIC_1013.OP'
          - '02FI_1000.PV'
        TimeStamp may be the index OR a column named `timestamp_col`.
        Should contain at least `roc_window` minutes of history so the rate of
        change can be computed. The LAST row is treated as "now".
    algo : ControlActionAlgorithm, optional
        Algorithm instance. A default one is created if None.
    roc_window : int
        Minutes of history used to estimate rate of change.
    timestamp_col : str
        Name of the timestamp column if TimeStamp is not already the index.

    Returns
    -------
    (AlgorithmOutput, PlantState, dict)
        output  : the algorithm result (severity, time_to_alarm, actions)
        state   : the PlantState built from the final row
        current_ops : {tag: current OP value} for every action tag present in df
                      (useful to compute concrete target OPs for execution)
    """
    import pandas as pd

    if algo is None:
        algo = ControlActionAlgorithm()

    data = df.copy()

    # Accept OTS-style underscore column names (e.g. 03LIC_1071_PV -> 03LIC_1071.PV)
    if not any(str(c).endswith('.PV') for c in data.columns):
        import re
        data.columns = [re.sub(r'_(PV|OP|SP|MODE)$', r'.\1', str(c)) for c in data.columns]

    # Normalise timestamp to a sorted DatetimeIndex
    if timestamp_col in data.columns:
        data[timestamp_col] = pd.to_datetime(data[timestamp_col])
        data = data.set_index(timestamp_col)
    else:
        data.index = pd.to_datetime(data.index)
    data = data.sort_index()

    if data.empty:
        raise ValueError("Snapshot dataframe is empty.")

    # "Now" = the last available timestamp (just before the alarm)
    now = data.index[-1]

    # Build the plant state from the snapshot
    state = build_state_from_data(data, now, roc_window=roc_window)

    # Run the algorithm (cause tag(s) come from the upstream RCA module)
    output = algo.compute_actions(state, rca_cause_tags=rca_cause_tags)

    # Collect current OP values for every lever tag we can see, so the caller
    # can turn the recommended step into a concrete target OP.
    current_ops = {}
    last_row = data.iloc[-1]
    for tag in algo.all_lever_tags():
        op_col = f'{tag}.OP'
        if op_col in data.columns and pd.notna(last_row.get(op_col)):
            current_ops[tag] = float(last_row[op_col])

    return output, state, current_ops


def format_snapshot_result(output, state, current_ops=None):
    """
    Human-readable summary of a snapshot run, including concrete target OPs
    when current OP values are available.
    """
    lines = []
    lines.append("=" * 68)
    lines.append(f"SNAPSHOT @ {state.timestamp}")
    lines.append("=" * 68)
    lines.append(f"  03LIC_1071.PV : {state.pv_1071:.2f}   (alarm limit {ALARM_THRESHOLD})")
    lines.append(f"  Rate of change: {state.roc_1071:+.3f} /min")
    def _fmt(v, p):
        return f"{v:.{p}f}" if v is not None else "n/a"
    lines.append(f"  03PIC_1013.PV : {_fmt(state.pv_pic_1013, 1)}   OP: {_fmt(state.op_pic_1013, 1)}")
    lines.append(f"  02FI_1000.PV  : {_fmt(state.fi_1000, 3)}")
    lines.append("")
    lines.append(f"  Severity      : {output.severity.name}")
    t = output.time_to_alarm_minutes
    t_str = "inf (not falling)" if t == float('inf') else f"{t:.1f} min"
    lines.append(f"  Time to alarm : {t_str}")
    lines.append(f"  Diagnosis     : {output.diagnosis}")
    lines.append("")

    if not output.actions:
        lines.append("  RECOMMENDED ACTIONS: none (monitor only)")
        return '\n'.join(lines)

    lines.append(f"  RECOMMENDED ACTIONS ({len(output.actions)}):")
    current_ops = current_ops or {}
    for a in output.actions:
        var = a.action_type.value  # 'OP' or 'SP'
        prefix = "MODE->MANUAL, " if a.mode_to_manual else ""
        if a.target_value is not None:
            tgt_str = f"  [{var} -> {a.target_value:.1f}]"
        else:
            cur = current_ops.get(a.tag)
            if cur is not None and a.action_type == ActionType.OP_CHANGE:
                limits = OP_LIMITS.get(a.tag, {'min': 0.0, 'max': 100.0})
                target = max(limits['min'], min(limits['max'], cur + a.step))
                tgt_str = f"  [{cur:.1f} -> {target:.1f}]"
            else:
                tgt_str = ""
        lines.append(f"    #{a.priority} {prefix}{a.tag}.{var} {a.direction} "
                     f"by {a.magnitude:.1f}{tgt_str}")
        lines.append(f"        wait {a.response_lag_minutes} min - {a.reason}")
    return '\n'.join(lines)


# ==============================================================================
# EPISODE REPLAY — validate algorithm against historical episodes
# ==============================================================================

def replay_episode(pv_data, episode_start, episode_end, 
                   trigger_offset_minutes=10,
                   algo=None,
                   scan_interval_minutes=1,
                   rca_cause_tags=None):
    """
    Replay an alarm episode: trigger algorithm at a specified time before/after alarm
    and collect its recommendations over the episode duration.
    
    Parameters
    ----------
    pv_data : pd.DataFrame
        Full PV/OP time series (TimeStamp indexed).
    episode_start : pd.Timestamp
        Alarm start time.
    episode_end : pd.Timestamp
        Alarm end time.
    trigger_offset_minutes : int
        How many minutes before alarm start to trigger algorithm.
        Positive = before alarm (prevention mode).
        Negative = after alarm (resolution mode, trigger is |offset| min into alarm).
    algo : ControlActionAlgorithm, optional
        Algorithm instance. Uses default if None.
    scan_interval_minutes : int
        How often to run the algorithm scan.
    
    Returns
    -------
    list of AlgorithmOutput for each scan in the replay window.
    """
    import pandas as pd
    
    if algo is None:
        algo = ControlActionAlgorithm()
    
    trigger_time = episode_start - pd.Timedelta(minutes=trigger_offset_minutes)
    replay_end = episode_end + pd.Timedelta(minutes=10)  # Continue 10 min past alarm end
    
    # Need some history before trigger for rate calculation
    data_start = trigger_time - pd.Timedelta(minutes=10)
    
    # Slice data
    episode_data = pv_data.loc[data_start:replay_end].copy()
    
    if episode_data.empty:
        return []
    
    results = []
    current_time = trigger_time
    
    while current_time <= replay_end:
        if current_time not in episode_data.index:
            # Find nearest
            idx = episode_data.index.get_indexer([current_time], method='nearest')[0]
            if idx < 0 or idx >= len(episode_data):
                current_time += pd.Timedelta(minutes=scan_interval_minutes)
                continue
            current_time = episode_data.index[idx]
        
        state = build_state_from_data(episode_data, current_time)
        output = algo.compute_actions(state, rca_cause_tags=rca_cause_tags)
        results.append(output)
        
        current_time += pd.Timedelta(minutes=scan_interval_minutes)
    
    return results


def format_output(output: AlgorithmOutput) -> str:
    """Format an AlgorithmOutput as a human-readable string."""
    lines = []
    lines.append(f"[{output.timestamp}] Severity: {output.severity.name} | "
                 f"T_alarm: {output.time_to_alarm_minutes:.1f} min")
    lines.append(f"  Diagnosis: {output.diagnosis}")
    
    if output.actions:
        lines.append(f"  Actions ({len(output.actions)}):")
        for a in output.actions:
            var = a.action_type.value
            prefix = "MODE->MANUAL " if a.mode_to_manual else ""
            lines.append(f"    #{a.priority} {prefix}{a.tag}.{var} {a.direction} by {a.magnitude:.1f} "
                        f"(wait {a.response_lag_minutes}min) - {a.reason}")
    else:
        lines.append("  Actions: NONE (monitor only)")
    
    return '\n'.join(lines)


# ==============================================================================
# MAIN — quick demo
# ==============================================================================

# ==============================================================================
# REAL-TIME EXECUTION CONTROLLER
# ==============================================================================

class ActionExecutionController:
    """
    Stateful controller that manages action execution with cooldown logic.
    
    This is what would run on the OTS in real-time. It wraps the stateless
    algorithm and adds:
    1. Cooldown tracking per tag (don't re-act within response_lag window)
    2. Cumulative action tracking (know total OP change applied per tag)
    3. Escalation logic (if previous action didn't help, increase magnitude)
    4. De-escalation (when level recovers, start unwinding actions)
    
    Usage:
        controller = ActionExecutionController()
        
        # Every scan cycle (1 min):
        state = build_state_from_data(current_data, now)
        commands = controller.scan(state)
        for cmd in commands:
            # Execute cmd.tag MODE → MANUAL if needed
            # Set cmd.tag OP to cmd.target_op
            execute_on_ots(cmd)
    """
    
    @dataclass
    class TagState:
        """Track state of a tag we've acted on."""
        last_action_time: object = None  # pd.Timestamp
        cumulative_step: float = 0.0     # Total OP change we've applied
        original_op: Optional[float] = None  # OP before we first touched it
        mode_changed: bool = False       # Did we switch it to MANUAL?
        n_actions: int = 0               # How many times we've acted
    
    @dataclass  
    class ExecutionCommand:
        """A concrete command to execute on the OTS/DCS."""
        tag: str
        mode_to_manual: bool  # If True, switch to MANUAL first
        target_op: float      # Set OP to this value
        step: float          # Signed change from current OP
        reason: str
    
    def __init__(self, algo: Optional[ControlActionAlgorithm] = None,
                 current_op_values: Optional[dict] = None,
                 rca_cause_tags: Optional[list] = None):
        """
        Parameters
        ----------
        algo : ControlActionAlgorithm, optional
        current_op_values : dict, optional
            Initial OP values for tags we might act on.
            Keys: tag name (e.g., '03LIC_1071'), Values: current OP (float)
        rca_cause_tags : list, optional
            Cause tag(s) from the upstream RCA module, forwarded to the algorithm.
        """
        self.algo = algo or ControlActionAlgorithm()
        self.tag_states: dict = {}  # tag -> TagState
        self.current_ops: dict = current_op_values or {}
        self.rca_cause_tags = rca_cause_tags
        self.scan_count = 0
        self.active = False  # Are we currently intervening?
        self.recovery_detected = False
    
    def scan(self, state: PlantState) -> list:
        """
        Run one scan cycle. Returns list of ExecutionCommands to apply.
        
        The controller decides:
        - Which algorithm recommendations to actually execute (respecting cooldowns)
        - Whether to escalate (previous actions weren't enough)
        - Whether to start unwinding (recovery detected)
        """
        self.scan_count += 1
        
        # Get algorithm recommendation (cause tag(s) from RCA)
        output = self.algo.compute_actions(state, rca_cause_tags=self.rca_cause_tags)
        
        # Check for recovery
        if state.roc_1071 > 0.1 and state.pv_1071 > ALARM_THRESHOLD:
            if self.active:
                self.recovery_detected = True
        
        # If no actions recommended and we're recovering, consider unwinding
        if output.severity == Severity.NONE and self.recovery_detected:
            return self._unwind_actions(state)
        
        if not output.actions:
            return []
        
        # Filter actions by cooldown
        commands = []
        for action in output.actions:
            tag = action.tag

            # This controller executes OP moves only; SP recommendations are
            # surfaced by the stateless API and handled separately (out of scope).
            if action.action_type == ActionType.SP_CHANGE:
                continue

            # Initialize tag state if first time
            if tag not in self.tag_states:
                self.tag_states[tag] = self.TagState()
                # Record original OP if we know it
                if tag in self.current_ops:
                    self.tag_states[tag].original_op = self.current_ops[tag]
            
            ts = self.tag_states[tag]
            
            # Check cooldown: don't re-act within response lag
            if ts.last_action_time is not None:
                elapsed = (state.timestamp - ts.last_action_time).total_seconds() / 60
                if elapsed < action.response_lag_minutes:
                    continue  # Still waiting for previous action to take effect
            
            # Compute target OP
            current_op = self.current_ops.get(tag)
            if current_op is None:
                # Don't know current OP — just recommend the step
                step = action.step
                target_op = None
            else:
                step = action.step
                target_op = current_op + step
                # Clamp to limits
                limits = OP_LIMITS.get(tag, {'min': 0, 'max': 100})
                target_op = max(limits['min'], min(limits['max'], target_op))
                step = target_op - current_op
            
            if abs(step) < 0.5:
                continue  # Too small to bother
            
            # Create execution command
            cmd = self.ExecutionCommand(
                tag=tag,
                mode_to_manual=not ts.mode_changed,  # Only on first action
                target_op=target_op if target_op is not None else step,
                step=step,
                reason=action.reason,
            )
            commands.append(cmd)
            
            # Update tag state
            ts.last_action_time = state.timestamp
            ts.cumulative_step += step
            ts.n_actions += 1
            ts.mode_changed = True
            if target_op is not None:
                self.current_ops[tag] = target_op
            
            self.active = True
        
        return commands
    
    def _unwind_actions(self, state: PlantState) -> list:
        """
        When level recovers, gradually return tags to original OP values.
        
        Don't unwind all at once — do it in steps to avoid overcorrection.
        Unwind the tags with smallest cumulative change first (least impactful).
        """
        commands = []
        
        if state.pv_1071 < OPERATING_LOW:
            # Not recovered enough to operating range — don't unwind yet
            return []
        
        # Sort tags by absolute cumulative step (unwind smallest first)
        sorted_tags = sorted(self.tag_states.items(), 
                           key=lambda x: abs(x[1].cumulative_step))
        
        for tag, ts in sorted_tags:
            if ts.original_op is None or ts.cumulative_step == 0:
                continue
            
            current = self.current_ops.get(tag)
            if current is None:
                continue
            
            # Unwind by 25% of cumulative step per scan (gradual)
            unwind_step = -ts.cumulative_step * 0.25
            target = current + unwind_step
            
            # Don't overshoot original
            if ts.cumulative_step > 0:
                target = max(target, ts.original_op)
            else:
                target = min(target, ts.original_op)
            
            actual_step = target - current
            if abs(actual_step) < 0.5:
                # Close enough to original — mark as done
                ts.cumulative_step = 0
                continue
            
            cmd = self.ExecutionCommand(
                tag=tag,
                mode_to_manual=False,
                target_op=target,
                step=actual_step,
                reason=f"Unwinding: returning towards original OP ({ts.original_op:.1f})"
            )
            commands.append(cmd)
            
            self.current_ops[tag] = target
            ts.cumulative_step += actual_step
        
        # If all tags unwound, reset state
        if all(ts.cumulative_step == 0 for ts in self.tag_states.values()):
            self.active = False
            self.recovery_detected = False
            self.tag_states.clear()
        
        return commands
    
    def get_status(self) -> dict:
        """Current controller status summary."""
        return {
            'active': self.active,
            'recovery_detected': self.recovery_detected,
            'scan_count': self.scan_count,
            'tags_acted_on': {tag: {'cumulative_step': ts.cumulative_step, 
                                     'n_actions': ts.n_actions,
                                     'mode_changed': ts.mode_changed}
                             for tag, ts in self.tag_states.items()
                             if ts.n_actions > 0}
        }


if __name__ == '__main__':
    import pandas as pd
    
    print("=" * 70)
    print("Control Action Algorithm — 03LIC_1071 PVLO")
    print("=" * 70)
    
    # Load data
    pv_data = pd.read_parquet('DATA/PV-OP_data/03LIC_1071_JAN_2026.parquet')
    pv_data['TimeStamp'] = pd.to_datetime(pv_data['TimeStamp'])
    pv_data.set_index('TimeStamp', inplace=True)
    pv_data.sort_index(inplace=True)
    
    # Instantiate algorithm
    algo = ControlActionAlgorithm(aggressiveness=1.0)
    
    # Demo: Replay cluster 23 (episode 25) — gradual pressure-driven episode
    # Alarm start: 2022-02-09 19:13 (when PV crossed 28.75)
    episode_start = pd.Timestamp('2022-02-09 19:13:00')
    episode_end = pd.Timestamp('2022-02-09 19:29:00')
    
    print(f"\nReplaying episode: {episode_start} to {episode_end}")
    print(f"Triggering 10 min before alarm...\n")
    
    results = replay_episode(pv_data, episode_start, episode_end, 
                            trigger_offset_minutes=10, algo=algo)
    
    for r in results:
        print(format_output(r))
        print()
