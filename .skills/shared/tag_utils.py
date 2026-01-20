#!/usr/bin/env python3
"""
Shared tag utilities for Control Actions skills.

Provides common functions for working with tag names across different data sources.
"""

import pandas as pd
from typing import Set, List


def strings_similar(s1: str, s2: str) -> bool:
    """
    Check if two tag names are similar (handles naming variations).
    
    Rules:
    - First 3 characters must match exactly
    - Characters after underscore must match exactly OR have a 4+ char common substring
    
    Examples:
        strings_similar('03LIC_1071', '03LIC_1071') -> True
        strings_similar('03LCV_1071', '03LIC_1071') -> True (1071 matches)
        strings_similar('03PICA_1013', '03PIC_1013') -> True (1013 matches)
        strings_similar('03LIC_1071', '04LIC_1071') -> False (first 3 don't match)
    """
    s1 = str(s1).strip().replace(' ', '').upper()
    s2 = str(s2).strip().replace(' ', '').upper()
    
    # First 3 characters must match exactly
    if len(s1) < 3 or len(s2) < 3:
        return False
    if s1[:3] != s2[:3]:
        return False
    
    # Must have underscore
    if '_' not in s1 or '_' not in s2:
        return False
    
    s1_after = s1.split('_', 1)[1]
    s2_after = s2.split('_', 1)[1]
    
    # Check if exactly the same after underscore
    if s1_after == s2_after:
        return True
    
    # Check for common substring of at least 4 characters
    shorter = s1_after if len(s1_after) <= len(s2_after) else s2_after
    longer = s2_after if len(s1_after) <= len(s2_after) else s1_after
    
    for i in range(len(shorter) - 3):
        substring = shorter[i:i+4]
        if substring in longer:
            return True
    
    return False


def get_pv_op_pairs(df: pd.DataFrame) -> List[str]:
    """
    Get list of tags that have both PV and OP columns.
    
    These are controllable tags - tags where we can observe the process
    variable (PV) and also have a control output (OP) to adjust.
    
    Args:
        df: DataFrame with columns ending in .PV and .OP
    
    Returns:
        Sorted list of base tag names (without .PV/.OP suffix)
    """
    cols = df.columns.tolist()
    op_tags = {col.replace('.OP', '') for col in cols if col.endswith('.OP')}
    pv_tags = {col.replace('.PV', '') for col in cols if col.endswith('.PV')}
    return sorted(op_tags & pv_tags)


def get_controllable_tags(df: pd.DataFrame) -> List[str]:
    """Alias for get_pv_op_pairs - returns tags with both PV and OP."""
    return get_pv_op_pairs(df)


def get_pv_only_tags(df: pd.DataFrame) -> List[str]:
    """
    Get tags that have only PV columns (sensors/indicators, not controllable).
    
    These tags can be monitored but not directly controlled.
    """
    cols = df.columns.tolist()
    op_tags = {col.replace('.OP', '') for col in cols if col.endswith('.OP')}
    pv_tags = {col.replace('.PV', '') for col in cols if col.endswith('.PV')}
    return sorted(pv_tags - op_tags)


def get_op_only_tags(df: pd.DataFrame) -> List[str]:
    """
    Get tags that have only OP columns (unusual, may indicate data issue).
    """
    cols = df.columns.tolist()
    op_tags = {col.replace('.OP', '') for col in cols if col.endswith('.OP')}
    pv_tags = {col.replace('.PV', '') for col in cols if col.endswith('.PV')}
    return sorted(op_tags - pv_tags)


def categorize_tags(df: pd.DataFrame) -> dict:
    """
    Categorize all tags in a DataFrame.
    
    Returns:
        Dictionary with:
        - controllable: Tags with both PV and OP
        - pv_only: Tags with only PV (sensors)
        - op_only: Tags with only OP (unusual)
        - pv_columns: List of all .PV columns
        - op_columns: List of all .OP columns
        - other_columns: Non PV/OP columns
    """
    cols = df.columns.tolist()
    
    pv_cols = [col for col in cols if col.endswith('.PV')]
    op_cols = [col for col in cols if col.endswith('.OP')]
    other_cols = [col for col in cols if not col.endswith('.PV') and not col.endswith('.OP')]
    
    op_tags = {col.replace('.OP', '') for col in op_cols}
    pv_tags = {col.replace('.PV', '') for col in pv_cols}
    
    return {
        'controllable': sorted(op_tags & pv_tags),
        'pv_only': sorted(pv_tags - op_tags),
        'op_only': sorted(op_tags - pv_tags),
        'pv_columns': pv_cols,
        'op_columns': op_cols,
        'other_columns': other_cols
    }


def match_event_sources_to_ts_tags(event_sources: List[str], ts_tags: Set[str]) -> dict:
    """
    Map event Source values to time series tags using fuzzy matching.
    
    Args:
        event_sources: List of unique Source values from events data
        ts_tags: Set of base tag names from time series data
    
    Returns:
        Dictionary mapping event sources to matched TS tags
    """
    source_to_ts = {}
    unmatched = []
    
    for source in event_sources:
        matches = []
        for ts_tag in ts_tags:
            if strings_similar(source, ts_tag):
                matches.append(ts_tag)
        
        if matches:
            source_to_ts[source] = matches
        else:
            unmatched.append(source)
    
    return {
        'matched': source_to_ts,
        'unmatched': unmatched
    }


if __name__ == '__main__':
    # Quick tests
    print("Testing strings_similar()...")
    test_cases = [
        ('03LIC_1071', '03LIC_1071', True),
        ('03LCV_1071', '03LIC_1071', True),
        ('03PICA_1013', '03PIC_1013', True),
        ('03LIC_1071', '04LIC_1071', False),
        ('03LIC_1071', '03LIC_1016', False),
    ]
    
    for s1, s2, expected in test_cases:
        result = strings_similar(s1, s2)
        status = '✅' if result == expected else '❌'
        print(f"  {status} strings_similar('{s1}', '{s2}') = {result} (expected {expected})")
