#!/usr/bin/env python
"""
Library module that contains functions common to all our FARM-based programs for evaluation, training, application.
"""

import sys
import os.path
import datetime

from sklearn.metrics import confusion_matrix
import torch
import statistics
import numbers
import logging
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
from orderedattrdict import AttrDict
from argparse import ArgumentParser
import mlflow
import signal
import toml
import json
import farm.utils
import farm.infer
from farm.infer import Inferencer
import farm.modeling.tokenization
import farm.data_handler.processor
import farm.data_handler.data_silo
import farm.modeling.optimization
from farm.data_handler.data_silo import DataSiloForCrossVal, DataSiloForHoldout
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
# from farm.train import Trainer, EarlyStopping
from farm_tools.train_modified import Trainer, EarlyStopping
from farm_tools import train_modified
from farm_tools.farm_eval import OurEvaluator
from farm_tools import farm_eval
# from farm.eval import Evaluator
from farm.evaluation.metrics import registered_metrics
from farm_tools.utils import init_logger
from farm.visual.ascii.images import BUSH_SEP
from farm_tools.farm_tasks import *
from farm_tools.farm_optsched import *
from farm_tools.farm_utils import str2bool
logger = init_logger()


def install_signal_handler(mlf_logger):
    """
    Install the SIGINT signal handler which will log the "ABORT" parameter to the FARM MLFlowLogger
    (not directly to mlflow so we do NOT log if logging is disabled).

    :param mlf_logger: FARM MLFlowLogger instance
    :return:
    """

    def signal_handler(sig, frame):
        logger.error("Control-C / SIGINT received, aborting!!!!")
        mlf_logger.log_params({"ABORTED": "SIGINT"})
        sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)


DEFAULT_CONFIG_BASIC = AttrDict(dict(
    seed=42,
    n_gpu=1,
    use_cuda=None,
    use_amp=False,
    do_lower_case=False,
    text_column="text",
    batch_size=32,
    max_seq=64,
    deterministic="False"
))

DEFAULT_CONFIG_TRAIN = AttrDict(dict(
    label_column="target",
    dev_splt=0.1,
    dev_stratification=False,
    grad_acc=1,
    evaluate_every=10,
    max_epochs=20,
    dropout=0.2,
    lrate=0.5e-5,
    es_patience=10,
    es_metric="f1_micro",
    es_mode="max",
    es_min_evals=1,
    es_hd=0,
    losses_alpha=0.5,
    fts="FTSingleClassification",
    fts_cfg=None,  # multiple values of the form key=val
    fos="FOSDefault",
    fos_cfg=None,
    hd_dim=768,
    hd0_cfg=None,
    hd1_cfg=None,
    hd2_cfg=None,
    hd3_cfg=None,
    hd4_cfg=None,
    hd5_cfg=None,
    hd6_cfg=None,
    hd7_cfg=None,
    hd8_cfg=None,
    hd9_cfg=None,
))

DEFAULT_CONFIG_APPLY = AttrDict(dict(
    text_column="text",
    label_column="prediction",
    prob_column="prob",
    batch_size=32,
    max_seq=None,
    num_processes=1,
    n_gpu=1,   # NEEDED???
    do_lower_case=False,
))

DEFAULT_CONFIG_ESTIMATE = AttrDict(dict(
    eval_method="xval",  # possible values: xval, holdout, onfile
    xval_folds=10,
    holdout_repeats=5,
    holdout_train=0.7,
    eval_stratification=False,
    onfile_file="NEEDED"
))


DEFAULT_CONFIG_HSEARCH = AttrDict(dict(
    halg="grid",
    beamsize=3,
    halg_random_n=20,
    est_var="head0_f1_macro_mean",
    est_cmp="max",
))


def update_config(toupdate, updatewith):
    for k, v in updatewith.items():
        if v is not None:
            toupdate[k] = v
    return toupdate


def load_config(fpath):
    """
    Load configuration from fpath and return as AttrDict.

    :param fpath: configuration file path, either TOML or JSON file
    :return: configuration object
    """
    if fpath.endswith(".toml"):
        data = toml.load(fpath)
    elif fpath.endswith(".json"):
        with open(fpath, "rt", encoding="utf-8") as infp:
            data = json.load(infp)
    else:
        raise Exception(f"Cannot load config file {fpath}, must be .toml or json file")
    return AttrDict(data)


def getargs(parser, cfg):
    args = parser.parse_args()
    if args.cfg:
        cfg_add = load_config(args.cfg)
        update_config(cfg, cfg_add)
    update_config(cfg, vars(args))
    if cfg.get("labels") is None:
        cfg["label_list"] = ["0", "1"]
    else:
        cfg["label_list"] = cfg.labels.split(",")
    logger.info(f"Effective configuration: {cfg}")
    return cfg


def argparser_basic(parser=None, cfg=None, ignore_runname=False):
    """
    Creates the initial parser and config data structure for most applications.
    We extend the parser and the config with whatever additional bits we need before actually
    parsing the arguments.

    :return: a parser and a config data structure
    """
    if parser is None:
        parser = ArgumentParser()
    if cfg is None:
        cfg = AttrDict()
    DF = DEFAULT_CONFIG_BASIC
    cfg.update(DF)
    if not ignore_runname:
        parser.add_argument("--runname", type=str, required=True,
                            help="Experiment name. Files are stored in directory {runname}-{datetime}")
    parser.add_argument("--infile", type=str, required=True,
                        help="Path to input file")
    parser.add_argument("--cfg", type=str,
                        help="Path to configuration file")
    parser.add_argument("--seed", type=int,
                        help=f"Random seed ({DF.seed})")
    parser.add_argument("--n_gpu", type=int,
                        help=f"Number of GPUs, if GPU is to be used ({DF.n_gpu}")
    parser.add_argument("--use_cuda", default=None, type=str2bool,
                        help="If GPUs should be used, if not specified, determined from setup")
    parser.add_argument("--use_amp", type=str2bool,
                        help=f"Use AMP ({DF.use_amp}")
    parser.add_argument("--do_lower_case", type=str2bool,
                        help=f"Lower case tokens ({DF.do_lower_case})")
    parser.add_argument("--text_column", type=str,
                        help=f"Name of in/out text column ({DF.text_column})")
    parser.add_argument("--batch_size", type=int,
                        help=f"Batch size ({DF.batch_size})")
    parser.add_argument("--max_seq", type=int,
                        help=f"Maximum sequence length (whatever the trainer used)")
    parser.add_argument("--deterministic", type=str2bool, default="False",
                        help=f"Use deterministic (slower) code ({DF.deterministic})")
    parser.add_argument("-d", action="store_true", help="Enable debug mode")
    return parser, cfg


def argparser_estimate(parser=None, cfg=None):
    parser, cfg = argparser_train(parser, cfg)
    DF = DEFAULT_CONFIG_ESTIMATE
    if parser is None:
        parser = ArgumentParser()
    if cfg is None:
        cfg = AttrDict()
    cfg.update(DF)
    parser.add_argument("--eval_method", type=str,
                        help=f"Evaluation method, one of xval, holdout ({DF.eval_method})")
    parser.add_argument("--xval_folds", type=int,
                        help=f"Number of folds for xval ({DF.xval_folds})")
    parser.add_argument("--holdout_repeats", type=int,
                        help=f"Number of repetitions for holdout estimation ({DF.holdout_repeats})")
    parser.add_argument("--holdout_train", type=float,
                        help=f"Portion used for training for holdout estimation ({DF.holdout_train})")
    parser.add_argument("--eval_stratification", type=str2bool,
                        help=f"Use stratified samples for the evaluation splits? ({DF.eval_stratification})")
    return parser, cfg


def argparser_hsearch(parser=None, cfg=None):
    parser, cfg = argparser_estimate(parser, cfg)
    DF = DEFAULT_CONFIG_HSEARCH
    if parser is None:
        parser = ArgumentParser()
    if cfg is None:
        cfg = AttrDict()
    cfg.update(DF)
    parser.add_argument("--hcfg", type=str, required=True,
                        help="TOML configuration file for the hyperparameter search (required)")
    parser.add_argument("--outpref", type=str, required=True,
                        help=f"Output prefix for the files written for the hsearch run")
    parser.add_argument("--halg", type=str, default=DF.halg,
                        help=f"Search algorithm, one of grid, random, greedy, beam ({DF.halg})")
    parser.add_argument("--halg_rand_n", type=int, default=DF.halg_random_n,
                        help=f"Number of random runs for halg=random ({DF.halg_random_n})")
    parser.add_argument("--beamsize", type=str, default=DF.beamsize,
                        help=f"Size of beam for halg=beam ({DF.beamsize})")
    parser.add_argument("--est_var", type=str, default=DF.est_var,
                        help=f"Estimation variable to use for sorting/searching ({DF.est_var})")
    parser.add_argument("--est_cmp", type=str, default=DF.est_cmp,
                        help=f"Comparison to use for optimizing est_var, min or max ({DF.est_cmp})")
    return parser, cfg


def argparser_train(parser=None, cfg=None):
    parser, cfg = argparser_basic(parser, cfg)
    DF = DEFAULT_CONFIG_TRAIN
    if parser is None:
        parser = ArgumentParser()
    if cfg is None:
        cfg = AttrDict()
    cfg.update(DF)
    parser.add_argument("--label_column", type=str,
                        help=f"Name of label column ({DF.label_column})")
    parser.add_argument("--dev_splt", type=float,
                        help=f"Development set proportion ({DF.dev_splt})")
    parser.add_argument("--grad_acc", type=int,
                        help=f"Gradient accumulation steps ({DF.grad_acc})")
    parser.add_argument("--lm_dir", type=str,
                        help="Load LM from that directory instead of default")
    parser.add_argument("--lm_name", type=str,
                        help="Load LM from that known named model (will download and cache model!)")
    parser.add_argument("--evaluate_every", type=float,
                        help=f"Evaluate every this many batches ({DF.evaluate_every})")
    parser.add_argument("--max_epochs", type=int,
                        help=f"Maximum number of epochs ({DF.max_epochs})")
    parser.add_argument("--dropout", type=float,
                        help=f"Dropout rate ({DF.dropout})")
    parser.add_argument("--lrate", type=float,
                        help=f"Learning rate ({DF.lrate})")
    parser.add_argument("--es_patience", type=int,
                        help=f"Early stopping patience ({DF.es_patience})")
    parser.add_argument("--es_metric", type=str,
                        help=f"Early stopping metric ({DF.es_metric})")
    parser.add_argument("--es_mode", type=str,
                        help=f"Early stopping mode ({DF.es_mode})")
    parser.add_argument("--es_min_evals", type=int,
                        help=f"Early stopping minimum evaluation steps ({DF.es_min_evals})")
    parser.add_argument("--es_hd", type=int,
                        help=f"Early stopping head number to use ({DF.es_hd})")
    parser.add_argument("--labels", type=str, default=None,
                        help=f"Comma separated list of labels, if missing, assume '0' and '1'")
    parser.add_argument("--dev_stratification", type=str2bool,
                        help=f"Use stratified dev set splits? ({DF.dev_stratification})")
    parser.add_argument("--fts", type=str,
                        help=f"FarmTasks class to use ({DF.fts})")
    parser.add_argument("--fts_cfg", nargs='*', default=[],
                        help=f"FarmTasks configuration settings of the form parm=value")
    parser.add_argument("--fos", type=str,
                        help=f"FarmOptSched class to use ({DF.fos})")
    parser.add_argument("--fos_cfg", nargs='*', default=[],
                        help=f"Farm optimizer/scheduler configuration settings of the form parm=value")
    parser.add_argument("--hd_dim", type=int, default=DF.hd_dim,
                         help=f"Dimension of the LM output, i.e. the head input ({DF.hd_dim})")
    parser.add_argument("--hd0_cfg", nargs='*', default=[],
                        help=f"Head 0 config parameters of the form parm=value")
    parser.add_argument("--hd1_cfg", nargs='*', default=[],
                        help=f"Head 1 config parameters of the form parm=value")
    parser.add_argument("--hd2_cfg", nargs='*', default=[],
                        help=f"Head 2 config parameters of the form parm=value")
    parser.add_argument("--hd3_cfg", nargs='*', default=[],
                        help=f"Head 2 config parameters of the form parm=value")
    parser.add_argument("--hd4_cfg", nargs='*', default=[],
                        help=f"Head 2 config parameters of the form parm=value")
    parser.add_argument("--losses_alpha", type=float, default=DF.losses_alpha,
                        help=f"Alpha for loss aggregation (weight of head 0, weight for head 1 is 1-alpha)")
    return parser, cfg


def argparser_apply(parser=None, cfg=None):
    parser, cfg = argparser_basic(parser, cfg, ignore_runname=True)
    DF = DEFAULT_CONFIG_APPLY
    if parser is None:
        parser = ArgumentParser()
    if cfg is None:
        cfg = AttrDict()
    cfg.update(DF)
    parser.add_argument("--outfile", type=str, required=True,
                        help="Path to output TSV file")
    parser.add_argument("--modeldir", type=str, required=True,
                        help="Path to directory where the model is stored")
    parser.add_argument("--heads", type=int, nargs="*", 
                        help="Head numbers to use, prepends hdN to output column names")
    # TODO: these should come from the task name maybe?
    parser.add_argument("--label_column", type=str,
                        help=f"Name of added label column ({DF.label_column})")
    parser.add_argument("--prob_column", type=str,
                        help=f"Name of added probability column ({DF.prob_column})")
    parser.add_argument("--num_processes", default=None,
                        help=f"Number of processes to use ({DF.num_processes})")
    return parser, cfg


def init_farm(cfg, logger=logger):
    if cfg.get("use_cuda") is None:
        use_cuda = torch.cuda.is_available()
    else:
        use_cuda = cfg.use_cuda
    device, n_gpu = farm.utils.initialize_device_settings(use_cuda=use_cuda)
    if cfg.get("deterministic", False):
        farm.utils.set_all_seeds(seed=cfg.get("seed", 41), deterministic_cudnn=True)
        # torch.set_deterministic(True)
        torch.use_deterministic_algorithms(True)
    else:
        farm.utils.set_all_seeds(seed=cfg.get("seed", 41), deterministic_cudnn=False)
        # torch.set_deterministic(False)
        torch.use_deterministic_algorithms(False)
    device = str(device)  # this should give cuda or cpu
    if use_cuda:
        n_gpu = max(n_gpu, cfg.n_gpu)
    cfg.n_gpu = n_gpu
    cfg.device = device
    cfg.cuda_used = use_cuda
    logger.info("Device={}, nGPU={}".format(device, n_gpu))
    mlflow.log_params({"device": str(device)})
    mlflow.log_params({"n_gpu": str(n_gpu)})
    train_modified.looger = logger
    farm_eval.logger = logger
    # farm.train.logger = logger
    # farm.eval.logger = logger
    farm.utils.logger = logger
    farm.infer.logger = logger


def log_results(results, name, steps=None, logging=True, print=True, num_fold=None):
    assert steps is not None or num_fold is not None
    use_steps = steps or num_fold
    # Print a header
    header = "\n\n"
    header += BUSH_SEP + "\n"
    header += "***************************************************\n"
    if num_fold is not None:
        header += f"***** EVALUATION {name} | FOLD: {num_fold} *****\n"
    else:
        header += f"***** EVALUATION {name} *****\n"
    header += "***************************************************\n"
    header += BUSH_SEP + "\n"
    logger.info(header)

    for head_num, head in enumerate(results):
        taskname = head["task_name"]
        logger.info(f"\n _________ head {head_num} of {len(results)}: {taskname} _________")
        for metric_name, metric_val in head.items():
            # log with ML framework (e.g. Mlflow)
            if logging:
                if not metric_name in ["preds", "labels"] and not metric_name.startswith("_"):
                    if isinstance(metric_val, numbers.Number):
                        mlflow.log_metrics(
                            metrics={
                                f"{name}_{metric_name}_{head['task_name']}": metric_val
                            },
                            step=use_steps,
                        )
            # print via standard python logger
            if print:
                if metric_name == "report":
                    if isinstance(metric_val, str) and len(metric_val) > 8000:
                        metric_val = metric_val[:7500] + "\n ............................. \n" + metric_val[-500:]
                    logger.info("{}: \n {}".format(metric_name, metric_val))
                else:
                    if not metric_name in ["preds", "labels"] and not metric_name.startswith("_"):
                        logger.info("{} {}: {}".format(taskname, metric_name, metric_val))


def train_model(silo, save_dir, lang_model_dir, cfg=None):
    language_model = LanguageModel.load(lang_model_dir)

    # TODO: use our own task classes here to create one or more prediction heads to use
    ft = cfg["_fts"]   # the actual farm tasks instance (not the name)
    prediction_heads = ft.get_heads(silo)

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=prediction_heads,
        embeds_dropout_prob=cfg.dropout,
        lm_output_types=["per_sequence"] * len(prediction_heads),
        loss_aggregation_fn=ft.get_loss_aggregation_fn(silo),
        device=cfg.device)

    logger.info(f"Model used for training:\n{model}")
    logger.info(f"Number of named model parameters: {len(list(model.named_parameters()))}")
    logger.info(f"Number of all model parameters: {len(list(model.parameters()))}")
    if cfg.d:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"PARAMETER name={name}, shape={param.data.shape}")
            else:
                logger.info(f"NOGRAD: name={name}, shape={param.data.shape}")
    # Create an optimizer, this was the original code
    # model, optimizer, lr_schedule = initialize_optimizer(
    #     model=model,
    #     learning_rate=cfg.lrate,
    #     device=cfg.device,
    #     n_batches=len(silo.loaders["train"]),
    #     n_epochs=cfg.max_epochs,
    #     use_amp=cfg.use_amp,
    #     optimizer_opts=None,
    #     schedule_opts={"name": "CosineWarmupWithRestarts", "warmup_proportion": 0.4},
    #     grad_acc_steps=cfg.grad_acc,
    # )
    # use our own optimizer initializer instead:
    logger.info("Create optimizer/scheduler")
    fosname = cfg["fos"]
    clazz = globals().get(fosname)
    if clazz is None:
        raise Exception(f"FarmOptSched class {fosname} unknown")
    fos = clazz(
        model=model,
        n_batches=len(silo.loaders["train"]),
        n_epochs=cfg.max_epochs,
        device=cfg.device,
        learning_rate=cfg.lrate,
        grad_acc_steps=cfg.grad_acc,
        cfg=cfg
    )
    logger.info(f"Using Farm OptSched Instance: {fos} of type {type(fos)}")
    model, optimizer, lr_schedule = fos.get_optsched()
    if cfg.d:
        logger.info(f"Created optimizer: {optimizer}")
    logger.info(f"Created scheduler: {lr_schedule}")
    earlystopping = EarlyStopping(
        head=cfg.es_hd,
        metric=cfg.es_metric,
        mode=cfg.es_mode,
        min_evals=cfg.es_min_evals,
        save_dir=os.path.join(save_dir),  # where to save the best model
        patience=cfg.es_patience  # number of evaluations to wait for improvement before terminating the training
    )
    # if evaluate_every is < 0, interpret abs(evaluate_every) as number of epochs
    # for this we first need to find the number of batches per epoch
    eval_every = cfg.evaluate_every
    steps4epoch = len(silo.get_data_loader("train"))
    if eval_every < 0:
        nepochs = abs(eval_every)
        eval_every = int(nepochs * steps4epoch)
    else:
        eval_every = int(eval_every)
    neval4epoch = steps4epoch/eval_every
    logger.info(f"Evaluating every {eval_every} steps, {steps4epoch} steps, {neval4epoch} total per epoch")
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        data_silo=silo,
        epochs=cfg.max_epochs,
        n_gpu=cfg.n_gpu,
        lr_schedule=lr_schedule,
        evaluate_every=eval_every,
        device=cfg.device,
        grad_acc_steps=cfg.grad_acc,
        early_stopping=earlystopping,
        evaluator_test=False,
        disable_tqdm=True,
    )

    # train it
    trainer.train()
    return trainer.model


def run_xval(silo, save_dir, lang_model_dir, cfg):
    # Load one silo for each fold in our cross-validation
    save_dir = Path(save_dir)
    silos = DataSiloForCrossVal.make(silo,
                                     n_splits=cfg.xval_folds,
                                     stratification=cfg.eval_stratification,
                                     sets=["train", "dev"])
    for silo in silos:
        sz_train = len(silo.data.get("train", 0))
        sz_dev = len(silo.data.get("dev", 0))
        sz_test = len(silo.data.get("test", 0))
        logger.info("XVAL SPLIT SET SIZE: {}+{}+{}={}".format(
            sz_train, sz_dev, sz_test, sz_train+sz_dev+sz_test
        ))

    # for each fold, run the whole training, earlystopping to get a model, then evaluate the model
    # on the test set of each fold
    # Remember all the results for overall metrics over all predictions of all folds and for averaging
    allresults = []
    all_preds = None   # for each head, the list of all predictions over all folds
    all_labels = None  # for each head the list of all labels over all folds
    all_preferred_metrics = []
    all_train_times = []
    all_eval_times = []
    save_dir_root = Path(save_dir)
    if not save_dir_root.exists():
        raise Exception("Model saving path must exist: {}".format(save_dir_root))
    if not save_dir_root.is_dir():
        raise Exception("Model saving path must be a directory: {}".format(save_dir_root))
    for num_fold, silo in enumerate(silos):
        save_to = save_dir_root.joinpath("fold{}".format(num_fold))
        if not save_to.exists():
            save_to.mkdir()
        mlflow.start_run(run_name=f"fold-{num_fold + 1}-of-{len(silos)}", nested=True)
        logger.info(f"############ Crossvalidation: Fold {num_fold + 1} of {len(silos)} ############")
        tmptime = time.perf_counter()
        model = train_model(silo, save_to, lang_model_dir, cfg=cfg)
        all_train_times.append(time.perf_counter()-tmptime)

        # do eval on test set here (and not in Trainer),
        #  so that we can easily store the actual preds and labels for a "global" eval across all folds.

        evaluator_test = OurEvaluator(
            data_loader=silo.get_data_loader("test"),
            tasks=silo.processor.tasks,
            device=cfg.device,
            pass_instids=True,
            outdir=save_dir
        )
        # evaluator_test = Evaluator(
        #     data_loader=silo.get_data_loader("test"),
        #     tasks=silo.processor.tasks,
        #     device=cfg.device,
        # )

        tmptime = time.perf_counter()
        result = evaluator_test.eval(model, return_preds_and_labels=True, foldnr=num_fold)
        all_eval_times.append(time.perf_counter() - tmptime)
        log_results(result, "Fold", num_fold=num_fold)

        # for now we just calculate the average over all preferred metrics for all heads to get
        # the value for each fold.
        metrics4heads = [h.metric for h in model.prediction_heads]
        # NOTE: the metrics we allow here are ONLY registered metrics which refer to metrics classes!
        # So we replace the name with the actual class instances
        metrics4heads = [registered_metrics[m] for m in metrics4heads]
        # now calculate the preferred metrics for each head
        metricvals4heads = [metrics4heads[i].preferred(r) for i, r in enumerate(result)]
        all_preferred_metrics.append(statistics.mean(metricvals4heads))

        allresults.append(result)

        if all_preds is None:
            all_preds = [[] for _ in result]
            all_labels = [[] for _ in result]
        for i, r in enumerate(result):
            all_preds[i].extend(r.get("preds"))
            all_labels[i].extend(r.get("labels"))

        if cfg.device == "cuda":
            logger.info("CUDA: trying to release memory, current {}".format(
                torch.cuda.memory_allocated() / 1024 / 1024
            ))
            logger.info("(before) CUDA memory allocated: {}".format(
                torch.cuda.memory_allocated() / 1024 / 1024))
            logger.info("(before) CUDA max memory allocated: {}".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024))
            logger.info("(before) CUDA memory cached: {}".format(
                torch.cuda.memory_cached() / 1024 / 1024))
            logger.info("(before) CUDA max memory cached: {}".format(
                torch.cuda.max_memory_cached() / 1024 / 1024))
            model.cpu()  # MAYBE NOT NECESSARY BUT NOT SURE
            torch.cuda.empty_cache()
            logger.info("(after) CUDA memory allocated: {}".format(
                torch.cuda.memory_allocated() / 1024 / 1024))
            logger.info("(after) CUDA max memory allocated: {}".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024))
            logger.info("(after) CUDA memory cached: {}".format(
                torch.cuda.memory_cached() / 1024 / 1024))
            logger.info("(after) CUDA max memory cached: {}".format(
                torch.cuda.max_memory_cached() / 1024 / 1024))
        with open(str(save_to.joinpath("results.json")), "wt") as fp:
            json.dump(result, fp)
        mlflow.end_run()
    # Save the per-fold results to json for a separate, more detailed analysis
    with open(str(save_dir_root.joinpath("results-perfold.json")), "wt") as fp:
        json.dump(allresults, fp)

    # find the fold with the best average preferred metric value
    best_fold_idx = np.argmax(all_preferred_metrics)
    logger.info(f"Best fold index: {best_fold_idx}")
    mlflow.log_params({"XVAL_BEST_FOLD_IDX": best_fold_idx})
    mlflow.log_params({"XVAL_BEST_FOLD_METRIC": all_preferred_metrics[best_fold_idx]})
    # the following is a list that contains one defaultdict(list) per head.
    # each defaultdict(list) will have all the values for that head and metric from allresults
    xval_metric_lists_per_head = [defaultdict(list) for _ in allresults[0]]
    for resultsperhead in allresults:
        assert len(xval_metric_lists_per_head) == len(resultsperhead)
        for i, res in enumerate(resultsperhead):
            for name in res.keys():
                if name not in ["preds", "labels"] and \
                        not name.startswith("_") and \
                        isinstance(res[name], numbers.Number):
                    xval_metric_lists_per_head[i][name].append(res[name])
    # now collapse each of the lists into its mean, and add a stdev and var metric
    xval_metric_per_head = [{} for _ in xval_metric_lists_per_head]
    for i, res in enumerate(xval_metric_lists_per_head):
        newres = xval_metric_per_head[i]
        newres["dirname"] = str(save_dir_root)
        # newres["report"] = allresults[0][i].get("report", None)
        newres["task_name"] = allresults[0][i].get("task_name", "UNKNOWN TASKNAME ???")
        newres["time_train_mean"] = statistics.mean(all_train_times)
        newres["time_eval_mean"] = statistics.mean(all_eval_times)
        newres["time_total"] = sum(all_train_times)+sum(all_eval_times)
        for name in res.keys():
            values = res[name]
            vmean = statistics.mean(values)
            newres[name+"_mean"] = vmean
            newres[name+"_min"] = min(values)
            newres[name+"_max"] = max(values)
            if len(values) > 1:
                vstdev = statistics.stdev(values)
                vvar = statistics.variance(values)
                newres[name + "_stdev"] = vstdev
                newres[name + "_var"] = vvar
    log_results(xval_metric_per_head, "XVAL", steps=0)
    # add the confusion matrices per head
    for i, d in enumerate(xval_metric_per_head):
        # automatically determine the label list for the head
        tmplabels = set()
        tmplabels.update(all_labels[i])
        tmplabels.update(all_preds[i])
        tmplabels = list(tmplabels)
        tmplabels.sort()
        conf_matrix = confusion_matrix(all_labels[i], all_preds[i], labels=tmplabels)
        conf_matrix = conf_matrix.tolist()
        d["confusion_matrix"] = conf_matrix
        d["confusion_labels"] = tmplabels
        # log overall confusions matrix
        logger.info(f"Confusion matrix for head {i}:")
        l = " ".join(tmplabels)
        logger.info(f"    {l}")
        for j, row in enumerate(conf_matrix):
            r = " ".join([str(tmp) for tmp in row])
            logger.info(f"{tmplabels[j]}  {r}")
    with open(str(save_dir_root.joinpath("results-all.json")), "wt") as fp:
        json.dump(xval_metric_per_head, fp)
    return xval_metric_per_head


def run_holdout(silo, save_dir, lang_model_dir, cfg):
    # Load one silo for each holdout repition
    save_dir = Path(save_dir)
    silos = DataSiloForHoldout.make(silo,
                                    n_splits=cfg.holdout_repeats,
                                    stratification=cfg.eval_stratification,
                                    random_state=cfg.seed,
                                    train_split=cfg.holdout_train,
                                    sets=["train", "dev"])
    for silo in silos:
        sz_train = len(silo.data.get("train", 0))
        sz_dev = len(silo.data.get("dev", 0))
        sz_test = len(silo.data.get("test", 0))
        logger.info("HOLDOUT SPLIT SET SIZE: train={} dev={} test={} all={}".format(
            sz_train, sz_dev, sz_test, sz_train+sz_dev+sz_test
        ))
        # tmp_train = silo.data.get("train")
        # tmp_test = silo.data.get("test")
        # tmp_dev = silo.data.get("dev")
        # logger.info(f"!!!!DEBUG first instance from train {list(tmp_train)[0]}")
        # logger.info(f"!!!!DEBUG last  instance from train {list(tmp_train)[-1]}")
        # logger.info(f"!!!!DEBUG first instance from test  {list(tmp_test)[0]}")
        # logger.info(f"!!!!DEBUG last  instance from test  {list(tmp_test)[-1]}")
        # logger.info(f"!!!!DEBUG first instance from dev   {list(tmp_dev)[0]}")
        # logger.info(f"!!!!DEBUG last  instance from dev   {list(tmp_dev)[-1]}")
    # for each repetition, run the whole training, earlystopping to get a model, then evaluate the model
    # on the test set of each fold
    # Remember all the results for overall metrics over all predictions of all folds and for averaging
    allresults = []
    all_preds = None   # for each head, the list of all predictions over all folds
    all_labels = None  # for each head the list of all labels over all folds
    all_preferred_metrics = []
    all_train_times = []
    all_eval_times = []
    save_dir_root = Path(save_dir)
    if not save_dir_root.exists():
        raise Exception("Model saving path must exist: {}".format(save_dir_root))
    if not save_dir_root.is_dir():
        raise Exception("Model saving path must be a directory: {}".format(save_dir_root))
    for num_fold, silo in enumerate(silos):
        save_to = save_dir_root.joinpath("fold{}".format(num_fold))
        if not save_to.exists():
            save_to.mkdir()
        mlflow.start_run(run_name=f"fold-{num_fold + 1}-of-{len(silos)}", nested=True)
        logger.info(f"############ Holdout estimation: Split {num_fold + 1} of {len(silos)} ############")
        tmptime = time.perf_counter()
        model = train_model(silo, save_to, lang_model_dir, cfg=cfg)
        all_train_times.append(time.perf_counter()-tmptime)

        # do eval on test set here (and not in Trainer),
        #  so that we can easily store the actual preds and labels for a "global" eval across all folds.

        evaluator_test = OurEvaluator(
            data_loader=silo.get_data_loader("test"),
            tasks=silo.processor.tasks,
            device=cfg.device,
            pass_instids=True,
            outdir=save_dir,
        )
        # evaluator_test = Evaluator(
        #     data_loader=silo.get_data_loader("test"),
        #     tasks=silo.processor.tasks,
        #     device=cfg.device,
        # )

        tmptime = time.perf_counter()
        result = evaluator_test.eval(model, return_preds_and_labels=True, foldnr=num_fold)
        all_eval_times.append(time.perf_counter()-tmptime)
        log_results(result, "Split", num_fold=num_fold)

        # for now we just calculate the average over all preferred metrics for all heads to get
        # the value for each fold.
        metrics4heads = [h.metric for h in model.prediction_heads]
        # NOTE: the metrics we allow here are ONLY registered metrics which refer to metrics classes!
        # So we replace the name with the actual class instances
        metrics4heads = [registered_metrics[m] for m in metrics4heads]
        # now calculate the preferred metrics for each head
        metricvals4heads = [metrics4heads[i].preferred(r) for i, r in enumerate(result)]
        all_preferred_metrics.append(statistics.mean(metricvals4heads))

        allresults.append(result)

        if all_preds is None:
            all_preds = [[] for _ in result]
            all_labels = [[] for _ in result]
        for i, r in enumerate(result):
            all_preds[i].extend(r.get("preds"))
            all_labels[i].extend(r.get("labels"))

        if cfg.device == "cuda":
            logger.info("CUDA: trying to release memory, current {}".format(
                torch.cuda.memory_allocated() / 1024 / 1024
            ))
            logger.info("(before) CUDA memory allocated: {}".format(
                torch.cuda.memory_allocated() / 1024 / 1024))
            logger.info("(before) CUDA max memory allocated: {}".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024))
            logger.info("(before) CUDA memory cached: {}".format(
                torch.cuda.memory_cached() / 1024 / 1024))
            logger.info("(before) CUDA max memory cached: {}".format(
                torch.cuda.max_memory_cached() / 1024 / 1024))
            model.cpu()  # MAYBE NOT NECESSARY BUT NOT SURE
            torch.cuda.empty_cache()
            logger.info("(after) CUDA memory allocated: {}".format(
                torch.cuda.memory_allocated() / 1024 / 1024))
            logger.info("(after) CUDA max memory allocated: {}".format(
                torch.cuda.max_memory_allocated() / 1024 / 1024))
            logger.info("(after) CUDA memory cached: {}".format(
                torch.cuda.memory_cached() / 1024 / 1024))
            logger.info("(after) CUDA max memory cached: {}".format(
                torch.cuda.max_memory_cached() / 1024 / 1024))
        with open(str(save_to.joinpath("results.json")), "wt") as fp:
            json.dump(result, fp)
        logger.info(f"Fold model and data saved to {save_to}")
        mlflow.end_run()
    # Save the per-fold results to json for a separate, more detailed analysis
    with open(str(save_dir_root.joinpath("results-persplit.json")), "wt") as fp:
        json.dump(allresults, fp)

    # find the fold with the best average preferred metric value
    best_fold_idx = np.argmax(all_preferred_metrics)
    logger.info(f"Best split index: {best_fold_idx}")
    mlflow.log_params({"HOLDOUT_BEST_SPLIT_IDX": best_fold_idx})
    mlflow.log_params({"HOLDOUT_BEST_SPLIT_METRIC": all_preferred_metrics[best_fold_idx]})
    # the following is a list that contains one defaultdict(list) per head.
    # each defaultdict(list) will have all the values for that head and metric from allresults
    eval_metric_lists_per_head = [defaultdict(list) for _ in allresults[0]]
    for resultsperhead in allresults:
        assert len(eval_metric_lists_per_head) == len(resultsperhead)
        for headnr, res in enumerate(resultsperhead):
            for name in res.keys():
                if name not in ["preds", "labels"] and \
                        not name.startswith("_") and \
                        isinstance(res[name], numbers.Number):
                    eval_metric_lists_per_head[headnr][name].append(res[name])
    # now collapse each of the lists into its mean, and add a stdev and var metric
    eval_metric_per_head = [{} for _ in eval_metric_lists_per_head]
    for i, res in enumerate(eval_metric_lists_per_head):
        newres = eval_metric_per_head[i]
        newres["dirname"] = str(save_dir_root)
        # newres["report"] = allresults[i][0].get("report", None)
        newres["task_name"] = allresults[0][i].get("task_name", "UNKNOWN TASKNAME ???")
        newres["time_train_mean"] = statistics.mean(all_train_times)
        newres["time_eval_mean"] = statistics.mean(all_eval_times)
        newres["time_total"] = sum(all_train_times)+sum(all_eval_times)
        for name in res.keys():
            values = res[name]
            vmean = statistics.mean(values)
            newres[name+"_mean"] = vmean
            newres[name+"_min"] = min(values)
            newres[name+"_max"] = max(values)
            if len(values) > 1:
                vstdev = statistics.stdev(values)
                vvar = statistics.variance(values)
                newres[name+"_stdev"] = vstdev
                newres[name+"_var"] = vvar
    log_results(eval_metric_per_head, "HOLDOUT", steps=0)
    # add the confusion matrices per head
    for i, d in enumerate(eval_metric_per_head):
        # automatically determine the label list for the head
        tmplabels = set()
        tmplabels.update(all_labels[i])
        tmplabels.update(all_preds[i])
        tmplabels = list(tmplabels)
        tmplabels.sort()
        conf_matrix = confusion_matrix(all_labels[i], all_preds[i], labels=tmplabels)
        conf_matrix = conf_matrix.tolist()
        d["confusion_matrix"] = conf_matrix
        d["confusion_labels"] = tmplabels
        # log overall confusions matrix
        logger.info(f"Confusion matrix for head {i}:")
        l = " ".join(tmplabels)
        logger.info(f"    {l}")
        for j, row in enumerate(conf_matrix):
            r = " ".join([str(tmp) for tmp in row])
            logger.info(f"{tmplabels[j]}  {r}")
    with open(str(save_dir_root.joinpath("results-all.json")), "wt") as fp:
        json.dump(eval_metric_per_head, fp)
    logger.info(f"Estimation data and folds saved to {save_dir_root}")
    return eval_metric_per_head


def run_estimate(cfg, logger=logger):
    if cfg.get("runname") is None:
        cfg["runname"] = "estimate"
    cfg.runname = cfg.runname + "_" + time.strftime('%Y%m%d_%H%M%S')
    savedir = cfg.runname

    logger.info(f"Running estimation with configuration: {cfg}")

    ml_logger = farm.utils.MLFlowLogger(tracking_uri=str(Path(savedir).joinpath("mlruns")))
    run_name = "eval"

    ml_logger.init_experiment(experiment_name="{} / {}".format(cfg.runname, str(datetime.datetime.now())[:19]), run_name=run_name)
    init_farm(cfg, logger=logger)

    ml_logger.log_params(cfg)

    logger.info("Experiment init")

    lang_model_dir = "/raid/data/models/bert/deepset_bert-base-german-cased"
    if cfg.get("lm_name"):
        lang_model_dir = cfg.lm_name
    if cfg.get("lm_dir"):
        lang_model_dir = cfg.lm_dir
    logger.info(f"Using language model directory: {lang_model_dir}")
    ml_logger.log_params({"use_lm_model": lang_model_dir})

    train_file = cfg.infile
    ml_logger.log_params({"train_file": train_file})
    logger.info(f"Using input file: {train_file}")

    #label_column_name = cfg.label_column
    #text_column_name = cfg.text_column

    max_seq_length = cfg.max_seq
    dev_split = cfg.dev_splt

    #label_list = cfg.label_list
    #ml_logger.log_params({"label_list": ",".join(label_list)})
    # Create tokenizer
    # Here we cannot just specify the model bin file, this requires a directory with all kinds of files
    # For now, test with the predefined name: bert-base-german-cased bert-base-german-dbmdz-cased
    # OK this downloads for
    # bert-base-german-cased: file https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt
    # bert-base-german-dbmdz-cased: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt
    # OK directly specifying the vocab file works
    logger.info(f"Load tokenizer from {lang_model_dir}")
    tokenizer = farm.modeling.tokenization.Tokenizer.load(
        pretrained_model_name_or_path=lang_model_dir,
        do_lower_case=cfg.do_lower_case)

    # register_metrics('mymetrics', ClassificationMetrics(label_list=label_list))

    logger.info("Create processor")
    ftname = cfg["fts"]
    clazz = globals().get(ftname)
    if clazz is None:
        raise Exception(f"FarmTasks class {ftname} unknown")
    ft = clazz(cfg)
    data_dir = os.path.dirname(train_file)
    if data_dir == "":
        data_dir = "."
    processor = ft.get_processor(
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        train_filename=os.path.basename(train_file),
        test_filename=None,
        dev_split=dev_split,
        dev_stratification=cfg.dev_stratification,
        data_dir=data_dir,
    )
    cfg["_fts"] = ft
    logger.info("Create data silo")
    silo = farm.data_handler.data_silo.DataSilo(
        processor=processor,
        max_processes=1,
        batch_size=cfg.batch_size)

    if cfg["eval_method"] == "xval":
        ret = run_xval(silo, savedir, lang_model_dir, cfg=cfg)
    elif cfg["eval_method"] == "holdout":
        ret = run_holdout(silo, savedir, lang_model_dir, cfg=cfg)
    else:
        raise Exception(f"Not supported: eval_method={cfg['eval_method']}")
    ml_logger.end_run()
    return ret


def run_apply(cfg, logger=logger):

    # logger.setLevel(logging.CRITICAL)  ## does not work
    if not cfg.d:
        logging.disable(logging.CRITICAL+1)
    init_farm(cfg, logger=logger)

    def process_output_batch(cfg, inferencer, batch, name2idx, outfp):
        """In-place modify batch: add label and prob columns at end"""
        dicts = [{"text": row[name2idx[cfg.text_column]]} for row in batch]
        ret = inferencer.inference_from_dicts(dicts)
        heads = cfg.get("heads", [0])
        # for each head we add 2 output columns (label, prob) in order 
        outcols = []
        for hdnr in heads:
            result = ret[hdnr][0]
            preds = result["predictions"]
            labels = [pred["label"] for pred in preds]
            probs = [pred["probability"] for pred in preds]
            assert len(batch) == len(labels)
            assert len(batch) == len(probs)
            outcols.append(labels)
            outcols.append(probs)
        for alldata in zip(batch, *outcols):
            incols = alldata[0]
            predcols = alldata[1:]
            print("\t".join(incols), "\t".join([str(x) for x in predcols]), sep="\t", file=outfp)

    logger.info("LOADING MODEL")
    if cfg.max_seq is None:
        inferencer = Inferencer.load(
            cfg.modeldir,
            batch_size=cfg.batch_size,
            gpu=cfg.cuda_used,
            return_class_probs=False,
            num_processes=cfg.num_processes,
            disable_tqdm=True,
        )
    else:
        inferencer = Inferencer.load(
            cfg.modeldir,
            batch_size=cfg.batch_size,
            max_seq_len=cfg.max_seq,
            gpu=cfg.cuda_used,
            return_class_probs=False,
            num_processes=cfg.num_processes,
            disable_tqdm=True,
        )

    used_max_seq_len = inferencer.processor.max_seq_len
    logging.info(f"Used max_seq_len is {used_max_seq_len}")
    mlflow.log_params({"used_max_seq_len": used_max_seq_len})
    inferencer.disable_tqdm = True

    # TODO: do we need to disable logging?
    with open(cfg.infile, "rt", encoding="utf-8") as infp:
        # read the header line which we always assume to exist
        cols = infp.readline().rstrip("\n\r").split("\t")
        name2idx = {n: i for i, n in enumerate(cols)}
        with open(cfg.outfile, "wt", encoding="utf8") as outfp:
            # write hdr
            outcols = cols.copy()
            heads = cfg.get("heads", [0])
            for hdnr in heads:
                outcols.append(f"hd{hdnr}_"+cfg.label_column)
                outcols.append(f"hd{hdnr}_"+cfg.prob_column)
            print("\t".join(outcols), file=outfp)
            batch = []  # batchsize rows to process
            for line in infp:
                fields = line.rstrip("\n\r").split("\t")
                batch.append(fields)
                if len(batch) >= cfg.batch_size:
                    process_output_batch(cfg, inferencer, batch, name2idx, outfp)
                    batch = []
            if len(batch) > 0:
                process_output_batch(cfg, inferencer, batch, name2idx, outfp)


def run_train(cfg, logger=logger):
    if cfg.get("runname") is None:
        cfg["runname"] = "train"
    cfg.runname = cfg.runname + "_" + time.strftime('%Y%m%d_%H%M%S')
    savedir = cfg.runname

    ml_logger = farm.utils.MLFlowLogger(tracking_uri=str(Path(savedir).joinpath("mlruns")))
    run_name = "train"

    ml_logger.init_experiment(experiment_name="{} / {}".format(cfg.runname, str(datetime.datetime.now())[:19]), run_name=run_name)
    init_farm(cfg, logger=logger)

    ml_logger.log_params(cfg)

    lang_model_dir = os.environ["HOME"] + "/models/bert/deepset_bert-base-german-cased"
    if cfg.get("lm_name"):
        lang_model_dir = cfg.lm_name
    if cfg.get("lm_dir"):
        lang_model_dir = cfg.lm_dir
    logger.info(f"Using language model directory: {lang_model_dir}")
    ml_logger.log_params({"use_lm_model": lang_model_dir})

    train_file = cfg.infile
    ml_logger.log_params({"train_file": train_file})
    logger.info(f"Using input file: {train_file}")

    # TODO: these should go into the experiment class
    #label_column_name = cfg.label_column
    #text_column_name = cfg.text_column

    max_seq_length = cfg.max_seq
    dev_split = cfg.dev_splt

    # TODO: Should go into the experiment class and get logged for each head, head0 to headk
    #label_list = cfg.label_list
    #ml_logger.log_params({"label_list": ",".join(label_list)})
    # Create tokenizer
    # Here we cannot just specify the model bin file, this requires a directory with all kinds of files
    # For now, test with the predefined name: bert-base-german-cased bert-base-german-dbmdz-cased
    # OK this downloads for
    # bert-base-german-cased: file https://int-deepset-models-bert.s3.eu-central-1.amazonaws.com/pytorch/bert-base-german-cased-vocab.txt
    # bert-base-german-dbmdz-cased: https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-vocab.txt
    # OK directly specifying the vocab file works
    logger.info(f"Load tokenizer from {lang_model_dir}")
    tokenizer = farm.modeling.tokenization.Tokenizer.load(
        pretrained_model_name_or_path=lang_model_dir,
        do_lower_case=cfg.do_lower_case)

    # TODO: we need a separate metric for each head
    #register_metrics('mymetrics', ClassificationMetrics(label_list=label_list))

    logger.info("Create processor")
    ftname = cfg["fts"]
    clazz = globals().get(ftname)
    if clazz is None:
        raise Exception(f"FarmTasks class {ftname} unknown")
    ft = clazz(cfg)
    data_dir=os.path.dirname(train_file)
    if data_dir == "":
        data_dir = "."
    processor = ft.get_processor(
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        train_filename=os.path.basename(train_file),
        test_filename=None,
        dev_split=dev_split,
        dev_stratification=cfg.dev_stratification,
        data_dir=data_dir,
    )
    if hasattr(ft, "label_list"):
        ml_logger.log_params({"label_list": ",".join(ft.label_list)})
    else:
        for i in range(ft.nheads):
            llist = getattr(ft, f"label_list{i}")
            ml_logger.log_params({f"label_list{i}": ",".join(llist)})
    cfg["_fts"] = ft  # the farm tasks object, stored with underscore-name to avoid logging!

    logger.info("Create data silo")
    silo = farm.data_handler.data_silo.DataSilo(
        processor=processor,
        batch_size=cfg.batch_size)

    model = train_model(silo, savedir, lang_model_dir, cfg=cfg)
    ml_logger.end_run()
    return model

