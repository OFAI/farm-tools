"""
Module with various useful functions to simplify analyzing experiment data in notebooks
"""
import os
import json
from glob import glob
import re
import plotly.graph_objects as pgo
from plotly.colors import DEFAULT_PLOTLY_COLORS
import pandas as pd

SHARED_KEYS = [
    "dirname", "time_total", "time_train_mean", "time_eval_mean",
    "aggregated_loss_mean", "aggregated_loss_stdev", "aggregated_loss_var", "aggregated_loss_min", "aggregated_loss_max",
    "instids_max", "instids_mean", "instids_min", "instids_stdev", "instids_var"
]
IGNORED_KEYS = ["confusion_matrix", "confusion_labels"]

def _merge_head_keys(data):
    """
    Expects a list of per-head dicts with metrics, returns a list with a single dict
    """
    newdata = {}
    for idx, hd in enumerate(data):
        for k, v in hd.items():
            if k in IGNORED_KEYS:
                continue
            if k in SHARED_KEYS:
                if idx == 0:
                    newdata[k] = v
            else:
                newdata[f"hd{idx}_{k}"] = v
    return [newdata]


def df4exps(dirpath, match=None, by_head=True):
    """
    Process all directories in dirpath which contain a results-all.json file and create a df with columns
    for each metric found in the first directory.

    If by_head is True, then an additional column indicates the head number and there is one row per head,
    if False, then all metrics are prefixxed by headN_ to indicate the head.

    Args:
        dirpath: the directory which contains the experiment directories
        match:  if not None, a regexp that must match the name of each experiment directory

    Returns:
        The dataframe with metrics per column and rows per directory or per directory and head.

    """
    # get all the subdirectories which have a results-all.json file in them
    dirs = [f for f in glob(os.path.join(dirpath, "*")) if os.path.isdir(f)]
    n_dirs = len(dirs)
    dirs = [d for d in dirs if os.path.exists(os.path.join(d, "results-all.json"))]
    n_dirs_results = len(dirs)
    if match:
        dirs = [d for d in dirs if re.match(match, d)]
    n_dirs_match = len(dirs)
    print(f"Number of directories: {n_dirs}, with results: {n_dirs_results}, matching: {n_dirs_match}")
    if len(dirs):
        dfdict = {}
        if by_head:
            dfdict["head"] = []
        datas = []
        # we need to go through the directories twice to be sure to collect all keys and all heads
        # the first time we simply add empty lists for each key we encounter
        for dirnr, dir in enumerate(dirs):
            with open(os.path.join(dir, "results-all.json")) as infp:
                data = json.load(infp)
            # data should be an array with one dict per head
            if not by_head:
                # create a single dict with all the metrics from all heads, prefixed properly
                # however, some keys are only kept once as they should be the same between heads: SHARED_KEYS
                data = _merge_head_keys(data)
            datas.append(data)

            for hd in data:
                for k in hd.keys():
                    if k not in IGNORED_KEYS and k not in dfdict:
                        dfdict[k] = []
        # end for 1

        for data in datas:
            for idx, hd in enumerate(data):
                if by_head:
                    dfdict["head"].append(idx)
                for k in dfdict:
                    if k != "head":
                        dfdict[k].append(hd.get(k))
        # end for 2
        # for k, v in dfdict.items():
        #     print(f"Key: {k}: {len(v)}")
        df = pd.DataFrame.from_dict(dfdict)
        return df
    else:
        return None


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

