#!/usr/bin/env python
"""
Debug loading of data into a silo on different systems: should get same instances, but we do not?

UPDATE: debug what the internal data formats really look like and possible ways to uniquely identify
the instances!
"""
import os
import farm.utils
import farm.modeling.tokenization
# from farm.data_handler.processor import TextClassificationProcessor
from farm_processor import OurTextClassificationProcessor
from farm.data_handler.data_silo import DataSilo
import random
SEED = 13
USE_CUDA = False
DIR=os.environ["HOME"]+"/models/bert/deepset_bert-base-german-cased"
COL="sexism_bin_max"
MAXSEQ=64
TRAINFILE = "rounds00to28_10.tsv"
TRAINDIR="data"

device, n_gpu = farm.utils.initialize_device_settings(use_cuda=USE_CUDA)
farm.utils.set_all_seeds(seed=SEED)
logger = farm.data_handler.data_silo.logger

logger.info("!!!!!!!!!!!!!! LOADING TOKENIZER")
tokenizer = farm.modeling.tokenization.Tokenizer.load(
        pretrained_model_name_or_path=DIR,
        do_lower_case=False)

logger.info("!!!!!!!!!!!!!! CREATING PROCESSOR")
processor = OurTextClassificationProcessor(
    tokenizer=tokenizer,
    max_seq_len=MAXSEQ,
    train_filename=TRAINFILE,
    test_filename=TRAINFILE,
    dev_split=0.0,
    #id_column_name="id",
    dev_stratification=False,
    data_dir=TRAINDIR,
    text_column_name="text")

logger.info("!!!!!!!!!!!!!! ADDING TASK")
processor.add_task(
    task_type="classification",
    name="task1",
    metric="acc",
    text_column_name="text",
    label_list=["0", "1"],
    label_column_name=COL
)

logger.info("!!!!!!!!!!!!!! CREATING SILO")
random.seed(SEED)
silo = DataSilo(
    max_processes=1,
    processor=processor,
    batch_size=2)
tmp_train = silo.data.get("train")
logger.info(f"Type of training data: {tmp_train}")
tmp_test = silo.data.get("test")
tmp_dev = silo.data.get("dev")
logger.info(f"!!!!DEBUG first instance from train {list(tmp_train)[0]}")
logger.info(f"!!!!DEBUG last  instance from train {list(tmp_train)[-1]}")

ldr_train = silo.get_data_loader("train")
logger.info(f"Type of training data loader: {tmp_train}")
b1 = next(iter(ldr_train))
logger.info(f"get first item from data loader: {b1}")

