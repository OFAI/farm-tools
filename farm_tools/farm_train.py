#!/usr/bin/env python
"""
BERT classification/ordinal regression training
"""

from farm_tools import farm_lib
from farm_tools.utils import init_logger

logger = init_logger()


def main():
    cfg = farm_lib.getargs(*farm_lib.argparser_train())
    farm_lib.run_train(cfg)


if __name__ == "__main__":
    main()
