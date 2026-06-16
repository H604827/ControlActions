"""
Control Action Algorithm for 03LIC_1071 PVLO Alarm Prevention & Resolution

First-principles approach for the OTS (Operator Training Simulator) scenario.

OTS Fault: "Feed Maximization" — increased feed overloads the propane refrigeration
section, causing pressure rise (PIC_1013 saturates), increased vaporization, and
level drop in vessel 3E107 (03LIC_1071).

Algorithm Design:
- Monitors 03LIC_1071.PV rate of change
- Severity scales with rate of fall and proximity to alarm threshold
- Actions: MODE → MANUAL + OP changes on selected tags
- Action magnitude proportional to severity
- Respects response lag: waits before re-evaluating after action

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
}


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


@dataclass
class PlantState:
    """Snapshot of current plant state at a single timestamp."""
    timestamp: object  # pd.Timestamp
    pv_1071: float     # Current level PV (%)
    op_1071: float     # Current level controller OP (%)
    pv_pic_1013: float  # Current pressure (kPa)
    op_pic_1013: float  # Current pressure controller OP (%)
    fi_1000: float     # Current feed rate

    # Rate of change (computed over rolling window)
    roc_1071: float = 0.0     # d(PV)/dt for level (per minute)
    roc_pic_1013: float = 0.0  # d(PV)/dt for pressure

    # Optional: other tags for richer context
    pv_lic_3178: Optional[float] = None
    op_lic_3178: Optional[float] = None
    pv_fic_3435: Optional[float] = None


@dataclass
class ControlAction:
    """A single control action to execute."""
    tag: str                    # Tag to act on (e.g., '03HIC_3100')
    action_type: ActionType     # MODE or OP
    value: float               # New value (for MODE: 1=MANUAL, 0=AUTO; for OP: target OP)
    direction: str             # 'increase' or 'decrease' (for OP)
    magnitude: float           # Absolute change in OP units
    reason: str                # Human-readable explanation
    priority: int              # Execution order (1 = first)
    response_lag_minutes: int  # Expected time before effect visible
    
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
                 action_tags: Optional[list] = None,
                 aggressiveness: float = 1.0):
        """
        Parameters
        ----------
        action_tags : list, optional
            Override default action tag set. Each entry is a dict with:
            {'tag': str, 'direction': str, 'tier': int, 'max_step': float, 'response_lag': int}
        aggressiveness : float
            Multiplier on action magnitude (1.0 = nominal, 0.5 = conservative, 2.0 = aggressive).
            Use for tuning on OTS.
        """
        self.aggressiveness = aggressiveness
        
        if action_tags is not None:
            self.action_tags = action_tags
        else:
            self.action_tags = self._default_action_tags()
    
    def _default_action_tags(self) -> list:
        """
        Default action tag configuration for the 'feed maximization' OTS fault.
        
        Tiered approach:
        - Tier 1: Not used (03LIC_1071 PID handles this itself)
        - Tier 2: Pressure path (03PIC_1013) — only if not saturated
        - Tier 3: Load reduction (HIC tags) — primary operator strategy
        - Tier 4: Retention/recirculation (LIC_3178, LIC_3153, FIC_3435)
        
        Direction logic (from ±30 min window analysis of 525 alarm clusters):
        - 'decrease' = reduce OP to reduce throughput/outflow
        - 'increase' = increase OP to increase retention/inflow
        
        max_step: Maximum single-scan OP change.
        Calibrated from median operator step sizes within ±30 min of alarm:
          - HIC tags: median step = 2.0, used as max_step (operators do multiple 2-unit steps)
          - FIC_3435: median step = 2.0
          - PIC_1013: median step = 2.0 (but many small 0.1 steps from APC)
          - PIC_3131: median decrease step = 2.0
          - LIC_3153/3178: median increase step = 2.0
        """
        return [
            # Tier 2: Pressure path
            # PIC_1013: 51% decrease pre-alarm (operators try to relieve pressure)
            # Post-alarm: 58% INCREASE — meaning operators RESTORE it after stabilization
            # → The alarm-response action is DECREASE (same as pre-alarm intent)
            # PIC_3131: 94% decrease pre-alarm, present in 33 clusters
            {'tag': '03PIC_3131', 'direction': 'decrease', 'tier': 2,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_PIC_1013,
             'reason': 'Reduce pressure (suction) to decrease propane vaporization'},
            {'tag': '03PIC_1013', 'direction': 'decrease', 'tier': 2,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_PIC_1013,
             'reason': 'Reduce discharge pressure control'},
            
            # Tier 3: Load reduction
            # These tags are acted on both pre-alarm AND post-alarm with consistent DECREASE
            # HIC_3100: 72% decrease pre, 71% decrease post — strongest and most consistent
            # HIC_1141: 73% decrease pre, 78% decrease post — very consistent
            # HIC_1151: 72% decrease pre, 51% decrease post — consistent pre, mixed post
            # 02HIC_1087: 71% decrease pre, 63% decrease post — consistent
            # 02HIC_1050: 64% decrease pre, 78% decrease post — consistent
            {'tag': '03HIC_3100', 'direction': 'decrease', 'tier': 3,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_LOAD_REDUCTION,
             'reason': 'Reduce plant load (present in 99 clusters, 72% decrease)'},
            {'tag': '03HIC_1141', 'direction': 'decrease', 'tier': 3,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_LOAD_REDUCTION,
             'reason': 'Reduce heater load (present in 58 clusters, 73% decrease)'},
            {'tag': '03HIC_1151', 'direction': 'decrease', 'tier': 3,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_LOAD_REDUCTION,
             'reason': 'Reduce throughput (present in 143 clusters, 72% decrease pre-alarm)'},
            {'tag': '02HIC_1087', 'direction': 'decrease', 'tier': 3,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_LOAD_REDUCTION,
             'reason': 'Reduce upstream feed rate (present in 40 clusters, 71% decrease)'},
            {'tag': '02HIC_1050', 'direction': 'decrease', 'tier': 3,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_LOAD_REDUCTION,
             'reason': 'Reduce upstream compressor load (present in 41 clusters, 78% decrease post)'},
            
            # Tier 4: Retention / recirculation
            # FIC_3435: 69% increase pre-alarm, 64% increase post — MOST FREQUENT tag (227 clusters)
            # LIC_3153: 79% increase pre, 81% increase post — very consistent direction
            # LIC_3178: 77% increase pre (small n=30), 70% increase post
            {'tag': '03FIC_3435', 'direction': 'increase', 'tier': 4,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_RETENTION,
             'reason': 'Increase propane recirculation (most acted tag, 227 clusters, 69% increase)'},
            {'tag': '03LIC_3153', 'direction': 'increase', 'tier': 4,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_RETENTION,
             'reason': 'Increase downstream liquid retention (81% increase post-alarm)'},
            {'tag': '03LIC_3178', 'direction': 'increase', 'tier': 4,
             'max_step': 2.0, 'response_lag': RESPONSE_LAG_RETENTION,
             'reason': 'Increase liquid retention in propane loop (70% increase post-alarm)'},
        ]

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
            # Already in alarm — severity based on how deep and how fast still falling
            if rate_magnitude > ROC_SEVERE:
                severity = Severity.CRITICAL
                diagnosis = f"IN ALARM. PV={pv:.1f} (below {ALARM_THRESHOLD}), still falling at {roc:.3f}/min"
            elif rate_magnitude > ROC_MODERATE:
                severity = Severity.SEVERE
                diagnosis = f"IN ALARM. PV={pv:.1f}, falling at {roc:.3f}/min"
            elif roc < -ROC_NOISE_FLOOR:
                severity = Severity.MODERATE
                diagnosis = f"IN ALARM. PV={pv:.1f}, slow fall at {roc:.3f}/min"
            else:
                severity = Severity.MILD
                diagnosis = f"IN ALARM but stabilizing. PV={pv:.1f}, roc={roc:.3f}/min"
        
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
        
        # Refine within band based on time_to_alarm
        if severity == Severity.CRITICAL and time_to_alarm <= 0:
            # Already in alarm — full power
            return 1.0
        elif time_to_alarm < 5 and time_to_alarm > 0:
            # Urgency boost within last 5 minutes
            return base + (1.0 - base) * (1 - time_to_alarm / 5)
        
        return base

    def select_action_tiers(self, severity: Severity, state: PlantState) -> list:
        """
        Determine which tiers of action to engage based on severity.
        
        Progressive engagement:
        - MILD: Monitor only (no actions, or very gentle Tier 4)
        - MODERATE: Tier 4 (retention) + Tier 2 if pressure not saturated
        - SEVERE: Tier 2 + 3 + 4
        - CRITICAL: All tiers, maximum response
        """
        tiers_to_engage = set()
        
        if severity == Severity.NONE:
            return []
        
        if severity.value >= Severity.MILD.value:
            tiers_to_engage.add(4)  # Retention — gentlest
        
        if severity.value >= Severity.MODERATE.value:
            # Add pressure path if PIC_1013 is not already saturated
            pic_headroom = state.op_pic_1013 - OP_LIMITS['03PIC_1013']['min']
            if pic_headroom > 5:  # At least 5% OP room to decrease
                tiers_to_engage.add(2)
        
        if severity.value >= Severity.SEVERE.value:
            tiers_to_engage.add(3)  # Load reduction — the big lever
            tiers_to_engage.add(2)  # Pressure regardless of saturation (try anyway)
        
        return sorted(tiers_to_engage)

    def compute_actions(self, state: PlantState) -> AlgorithmOutput:
        """
        Main algorithm entry point. Given current state, produce control actions.
        
        Parameters
        ----------
        state : PlantState
            Current snapshot of all relevant tag values and rates.
            
        Returns
        -------
        AlgorithmOutput with severity, time_to_alarm, and list of ControlActions.
        """
        # Step 1: Assess severity
        severity, time_to_alarm, diagnosis = self.assess_severity(state)
        
        output = AlgorithmOutput(
            timestamp=state.timestamp,
            severity=severity,
            time_to_alarm_minutes=time_to_alarm,
            diagnosis=diagnosis,
            actions=[]
        )
        
        if severity == Severity.NONE:
            return output
        
        # Step 2: Determine severity factor (0→1)
        sf = self.compute_severity_factor(severity, time_to_alarm)
        
        # Step 3: Select which tiers to engage
        tiers = self.select_action_tiers(severity, state)
        
        if not tiers:
            return output
        
        # Step 4: Compute action magnitude for each tag in engaged tiers
        priority = 1
        for tag_config in self.action_tags:
            if tag_config['tier'] not in tiers:
                continue
            
            tag = tag_config['tag']
            direction = tag_config['direction']
            max_step = tag_config['max_step']
            response_lag = tag_config['response_lag']
            reason = tag_config['reason']
            
            # Magnitude = severity_factor × max_step × aggressiveness
            # With max_step=2.0, at SEVERE (sf=0.75): 2.0 * 0.75 = 1.5
            # At CRITICAL (sf=1.0): 2.0 * 1.0 = 2.0 (one full operator step)
            # The aggressiveness multiplier allows multiple steps: 2.0 = two operator steps
            magnitude = sf * max_step * self.aggressiveness
            
            # Floor: below 0.5 is meaningless
            if magnitude < 0.5:
                continue
            
            # Clamp to max_step
            magnitude = min(magnitude, max_step)
            
            # Round to reasonable precision (0.5 units, as operators do)
            magnitude = round(magnitude * 2) / 2
            
            action = ControlAction(
                tag=tag,
                action_type=ActionType.OP_CHANGE,
                value=magnitude,  # The step SIZE (caller adds to current OP)
                direction=direction,
                magnitude=magnitude,
                reason=reason,
                priority=priority,
                response_lag_minutes=response_lag,
            )
            output.actions.append(action)
            priority += 1
        
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
    
    # Rate of change over window
    window_start = timestamp - np.timedelta64(roc_window, 'm')
    pv_window = pv_data_slice.loc[window_start:timestamp, '03LIC_1071.PV'].values
    roc_1071 = compute_rate_of_change(pv_window, window=roc_window, method='regression')
    
    pic_window = pv_data_slice.loc[window_start:timestamp, '03PIC_1013.PV'].values
    roc_pic_1013 = compute_rate_of_change(pic_window, window=roc_window, method='regression')
    
    state = PlantState(
        timestamp=timestamp,
        pv_1071=float(row['03LIC_1071.PV']),
        op_1071=float(row['03LIC_1071.OP']),
        pv_pic_1013=float(row['03PIC_1013.PV']),
        op_pic_1013=float(row['03PIC_1013.OP']),
        fi_1000=float(row['02FI_1000.PV']),
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
    
    return state


# ==============================================================================
# EPISODE REPLAY — validate algorithm against historical episodes
# ==============================================================================

def replay_episode(pv_data, episode_start, episode_end, 
                   trigger_offset_minutes=10,
                   algo=None,
                   scan_interval_minutes=1):
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
        output = algo.compute_actions(state)
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
            lines.append(f"    #{a.priority} {a.tag}.OP {a.direction} by {a.magnitude:.1f} "
                        f"(wait {a.response_lag_minutes}min) — {a.reason}")
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
                 current_op_values: Optional[dict] = None):
        """
        Parameters
        ----------
        algo : ControlActionAlgorithm, optional
        current_op_values : dict, optional
            Initial OP values for tags we might act on.
            Keys: tag name (e.g., '03HIC_3100'), Values: current OP (float)
        """
        self.algo = algo or ControlActionAlgorithm()
        self.tag_states: dict = {}  # tag -> TagState
        self.current_ops: dict = current_op_values or {}
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
        
        # Get algorithm recommendation
        output = self.algo.compute_actions(state)
        
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
