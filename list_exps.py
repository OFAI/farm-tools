#!/usr/bin/env python
"""
For each of all the subdirectories in a directory, check if it is an experimentation dir,
if yes, output selected information in one line of tsv or md table format
"""
import sys
import os
import json
import re
import argparse

# PAT_DIRNAME = re.compile(r'^([a-z-]+)([0-9]*)_([0-9]+)_([0-9]+)$')
PAT_NAME = re.compile(r'^([A-Za-z0-9-]+[a-z])([0-9]*)$')

def dir2data(dirpath):
    """
    Return all the metrics and data for a directory if it looks good, return None if it appears to be incomplete.

    :param dirpath: the path to the experiment directory
    :return: the result all dictionary, extended with the keys "runname", "rundate", "runtime", "dirname"
    """
    if dirpath.endswith("/"):
        dirpath = dirpath[:-1]
    rfile = os.path.join(dirpath, "results-all.json")
    if os.path.exists(rfile):
        with open(rfile, "rt") as infp:
            data = json.load(infp)   # this is an array, one dict for each head!
        dirname = os.path.basename(dirpath)
        for d in data:
            d["dirname"] = dirname
            runname = dirname[:-16]
            date = dirname[-15:-7]
            time = dirname[-6:]
            d["runname"] = runname
            d["rundate"] = date
            d["runtime"] = time
        return data
    else:
        return None


NAME2NAME = {
    "bin-base": ("Bin/B", 1),
    "bin-large": ("Bin/L", 2),
    "multi-base": ("Multi/B", 3),
    "multi-large": ("Multi/L", 4),
    "coral-base": ("Coral/B", 5),
    "coral-large": ("Coral/L", 6),
    "ordinal1-base": ("Ord/B", 6.2),
    "ordinal1-large": ("Ord/L", 6.5),
    "binmulti-base": ("BinMulti/B", 7),
    "binmulti-large": ("BinMulti/L", 8),
    "bincoral-base": ("BinCoral/B", 9),
    "bincoral-large": ("BinCoral/L", 10),
    "binordinal1-base": ("BinOrd/B", 10.5),
    "binordinal1-large": ("BinOrd/L", 11),
    
}


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Show experiment information for all subdirectories")
    parser.add_argument("--directory", type=str, default=".",
                        help="The containing directory, default is the current directory")
    parser.add_argument("--fmt", type=str, default="md",
                        help="Output format, one of 'tsv', 'latex' or 'md', default is 'md'")
    args = parser.parse_args()
    assert args.fmt in ["tsv", "md", "latex"]

    # for now we always output head, mean accuracy and stdev, mean f1macro and stdev
    #
    fnames = [os.path.join(args.directory, f) for f in os.listdir(args.directory)]
    subdirs = [f for f in fnames if os.path.isdir(f)]

    if args.fmt == "md":
        print("| Model | Date | Time | Head | Accuracy | Acc.Stdev | F1macro | F1m.Stdev | ")
        print("| ---- | ---- | ---- | ---- | --- | --------- | ----- | ----------- | ")
    elif args.fmt == "tsv":
        print("model", "date", "time", "head", "acc", "acc_stdev", "f1mac", "f1max_stdev", sep="\t")
    elif args.fmt == "latex":
        print("\\begin{tabular}{lrr}")
        print("\\toprule")
        print("Model& Accuracy & F1.0 macro \\\\")
        print("\\midrule")

    # read all data in then print in separate phase so we can influence the sort order
    alldatas = {}
    for dir in subdirs:
        data = dir2data(dir)
        if data is None:
            print(f"Warning: ignored directory {dir}, no complete data found", file=sys.stderr)
        else:
            name = data[0]["runname"]
            m = re.match(PAT_NAME, name)
            if m is None:
                print(f"Warning: ignoring directory {dir}, name {name} does not match")
                continue
            prefix, nr = m.groups()
            info = NAME2NAME[prefix]
            for d in data:
                d["model"] = info[0]
            alldatas[info[1]] = data

    # output the datas
    for idx in sorted(alldatas.keys()):
        data = alldatas[idx]
        nheads = len(data)
        for headnr, d in enumerate(data):
            name = d["runname"]
            date = d["rundate"]
            time = d["runtime"]

            if args.fmt == "md":
                print(f"| {name} | {date} | {time} | {headnr} | {d['acc_mean']:0.3f} | {d['acc_stdev']:0.3f} | {d['f1_macro_mean']:0.3f} | {d['f1_macro_stdev']:0.3f} | ")
            elif args.fmt == "tsv":
                print(f"{name}\t{date}\t{time}\t{headnr}\t{d['acc_mean']:0.3f}\t{d['acc_stdev']:0.3f}\t{d['f1_macro_mean']:0.3f}\t{d['f1_macro_stdev']:0.3f}")
            elif args.fmt == "latex":
                # replace some known model names for the paper
                name = name.replace("bincoral-", "BinCoral")
                name = name.replace("binmulti-", "BinMulti")
                name = name.replace("bin-", "Bin")
                name = name.replace("coral-", "Coral")
                name = name.replace("multi-", "Multi")
                name = name.replace("base01", "/B")
                name = name.replace("large01", "/L")
                if nheads == 2:
                    name += f":{headnr+1}"
                print(
                    f"{name} & {d['acc_mean']:0.3f}$\\pm${d['acc_stdev']:0.3f} & {d['f1_macro_mean']:0.3f}$\\pm${d['f1_macro_stdev']:0.3f} \\\\")

    if args.fmt == "latex":
        print("\\bottomrule")
        print("\\end{tabular}")

