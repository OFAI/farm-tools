#!/usr/bin/env python
"""
BERT classification/ordinal regression model estimation: builds one or more models and estimates performance
using xval, holdout (not yet) or testset estimation (not yet).

Currently supports classification and xval but should eventually also support ordinal regression,
multitask models, hold-out evaluation and more.
"""

import farm_lib
import mlflow
from utils import init_logger

logger = init_logger()

if __name__ == "__main__":

    cfg = farm_lib.getargs(*farm_lib.argparser_estimate())
    try:
        ret = farm_lib.run_estimate(cfg)
        logger.info(f"Got overall run_eval return value: {ret}")
    except Exception as ex:
        mlflow.log_params({"ABORTED": str(ex)})
        mlflow.end_run()
        raise ex
