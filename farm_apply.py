#!/usr/bin/env python
"""
BERT classification/ordinal regression application: apply the model to text from one column in the tsv input file,
output additional columns with label and probability
"""

import farm_lib
import mlflow
from gatenlp.utils import init_logger

logger = init_logger()

if __name__ == "__main__":

    cfg = farm_lib.getargs(*farm_lib.argparser_apply())
    farm_lib.run_apply(cfg)