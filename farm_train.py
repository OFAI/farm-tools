#!/usr/bin/env python
"""
BERT classification/ordinal regression training
"""

import farm_lib
from utils import init_logger

logger = init_logger()

if __name__ == "__main__":

    cfg = farm_lib.getargs(*farm_lib.argparser_train())
    farm_lib.run_train(cfg)
