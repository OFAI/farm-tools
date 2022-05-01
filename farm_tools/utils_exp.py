"""
Module with various useful functions to simplify analyzing experiment data in notebooks
"""
import plotly.graph_objects as pgo
from plotly.colors import DEFAULT_PLOTLY_COLORS
import pandas as pd


def add_labelcolumn(df, colname, newcolname, prefix="", f=None):
    """Create a new column that contains the value of the colname as a string, optionallu with a prefix"""
    if f is None:
        f = lambda x: x

    def conv(row):
        val = row[colname]
        if pd.isna(val):
            return f"{prefix}NaN"
        else:
            return f"{prefix}{f(row[colname])}"

    df[newcolname] = df.apply(lambda row: conv(row), axis=1)
    return df


def add_errorvals(df, col, lower, upper):
    """
    Given the columns for value, lower, upper value, calculate the values needed by plotly error and error minus.
    Use the original col name with _e and _em suffixes for that
    """
    df[col + "_e"] = df.apply(lambda row: row[upper] - row[col], axis=1)
    df[col + "_em"] = df.apply(lambda row: row[col] - row[lower], axis=1)
    return df


def traces_for(df, groupcol=None, x=None, y=None, sort=None, errorbars=False, stdev=None, **kwargs):
    """
    Return a list of barplot traces to add to a plotly figure where each trace is for one value
    in the column with name groupcol. x and y are the names of the x/y columns, if errorbars is True,
    tries to plot them based on x+"_e" and x+"_em", otherwise if stdev is not None, uses this as the
    name for plotting errorbars (e.g. stdev).
    """
    if sort is not None:
        df = df.sort_values(by=sort, axis=0, ascending=True)
    groupvals = sorted(list(df[groupcol].unique()))
    traces = []
    kwargs = kwargs.copy()
    for idx, gv in enumerate(groupvals):
        dftmp = df.loc[(df[groupcol] == gv)]
        if errorbars:
            kwargs["error_y"] = dict(array=dftmp[y + "_e"], arrayminus=dftmp[y + "_em"])
        elif stdev is not None:
            kwargs["error_y"] = dict(array=dftmp[stdev], arrayminus=dftmp[stdev])
        traces.append(pgo.Bar(name=gv, x=dftmp[x], y=dftmp[y],
                              marker=dict(color=DEFAULT_PLOTLY_COLORS[idx]), **kwargs))
    return traces

