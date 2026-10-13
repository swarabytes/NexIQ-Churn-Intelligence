"""
OLAP Engine — Olist Analytics
==============================
Provides reusable functions for Roll-Up, Drill-Down, Slice, and Dice
operations on the Olist analytical cube.
Completely isolated from the churn prediction pipeline.
"""

import pandas as pd


# ============================================================
# ROLL-UP
# ============================================================
def rollup(df, measure_cols, from_dims, to_dims):
    """
    Aggregate from a lower-level dimension to a higher-level dimension.

    Parameters
    ----------
    df : pd.DataFrame — The analytical base dataset.
    measure_cols : dict — {col: agg_func} e.g. {'order_id': 'nunique', 'total_sales': 'sum'}
    from_dims : list — Lower-level grouping columns.
    to_dims : list — Higher-level grouping columns (subset of from_dims or orthogonal).

    Returns
    -------
    lower_df, upper_df : two DataFrames at each aggregation level.
    """
    lower_df = df.groupby(from_dims).agg(measure_cols).reset_index()
    upper_df = df.groupby(to_dims).agg(measure_cols).reset_index()
    return lower_df, upper_df


# ============================================================
# DRILL-DOWN
# ============================================================
def drilldown(df, measure_cols, from_dims, to_dims, filter_col, filter_val):
    """
    Drill from a higher-level summary into a lower-level breakdown.

    Parameters
    ----------
    df : pd.DataFrame
    measure_cols : dict
    from_dims : list — Higher-level grouping.
    to_dims : list — Lower-level grouping.
    filter_col : str — Column to fix for the drill-down.
    filter_val : scalar — Value to select.

    Returns
    -------
    upper_row, lower_df : The summary row and its lower-level breakdown.
    """
    upper_df = df.groupby(from_dims).agg(measure_cols).reset_index()
    upper_row = upper_df[upper_df[filter_col] == filter_val]

    subset = df[df[filter_col] == filter_val]
    lower_df = subset.groupby(to_dims).agg(measure_cols).reset_index()
    return upper_row, lower_df


# ============================================================
# SLICE
# ============================================================
def slice_data(df, dimension, value):
    """
    Fix one dimension to a single value and return the filtered dataset.

    Parameters
    ----------
    df : pd.DataFrame
    dimension : str — Column to slice on.
    value : scalar — Selected value.

    Returns
    -------
    pd.DataFrame — Rows where dimension == value.
    """
    return df[df[dimension] == value].copy()


# ============================================================
# DICE
# ============================================================
def dice_data(df, filters):
    """
    Filter across multiple dimensions simultaneously.

    Parameters
    ----------
    df : pd.DataFrame
    filters : dict — {column: list_of_values} e.g.
              {'customer_state': ['SP', 'RJ'], 'year': [2017, 2018]}

    Returns
    -------
    pd.DataFrame — Rows satisfying ALL filter conditions.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    for col, values in filters.items():
        mask &= df[col].isin(values)
    return df[mask].copy()
