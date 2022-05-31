#!/usr/bin/env python
"""
BERT classification/ordinal regression application: apply the model to text from one column in the tsv input file,
output additional columns with label and probability
"""
from farm_tools import farm_lib
import farm_tools.farm_lib
import mlflow
from farm_tools.utils import init_logger

logger = init_logger()


def main():
    cfg = farm_lib.getargs(*farm_lib.argparser_apply())
    farm_lib.run_apply(cfg)


if __name__ == "__main__":
    main()
