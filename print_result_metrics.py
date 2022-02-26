#!/usr/bin/env python

import argparse
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="Name of the all results json file created by the estimation run")
    parser.add_argument("--metrics", required=True, nargs="+", help="One or more names of metrics to extract")
    parser.add_argument("--headnr", default=0, help="Head number to extract from, default is 0")
    args = parser.parse_args()

    assert args.metrics is not None and len(args.metrics) > 0
    with open(args.infile, "rt", encoding="utf-8") as infp:
        data = json.load(infp)
        hd_data = data[args.headnr]
        vals = [str(hd_data[n]) for n in args.metrics]
        print(args.infile, ",".join(vals))
