#!/usr/bin/env python
"""
Module to better test heads.
"""
import os
import torch
from torch.utils.data.sampler import SequentialSampler
from utils import init_logger
from farm_processor import OurTextClassificationProcessor
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.language_model import LanguageModel
from farm.data_handler.dataloader import NamedDataLoader

import farm.data_handler.data_silo
import farm.modeling.tokenization
logger = init_logger()

labels = ["0", "1", "2", "3", "4"]
device = "cpu"
# only linux!
lang_model_dir = os.environ["HOME"] + "/models/bert/deepset_bert-base-german-cased/"
train_file = os.environ["FEMDWELL"] + "/sexism-classification/corpora/rounds00to28_500.tsv"
batch_size = 2


def init_model(
        predictionhead,
        label_list=labels,
        label_column="sexism_orig_max",
        train_file=None,
):
    logger.info(f"Load tokenizer from {lang_model_dir}")
    tokenizer = farm.modeling.tokenization.Tokenizer.load(
        pretrained_model_name_or_path=lang_model_dir,
        do_lower_case=False)
    logger.info("Tokenizer loaded")
    logger.info(f"Create TextClassificationProcessor")
    processor = OurTextClassificationProcessor(
        text_column_name="text",
        tokenizer=tokenizer,
        max_seq_len=64,
        train_filename=os.path.basename(train_file),
        test_filename=None,
        dev_split=0.1,
        dev_stratification=True,
        data_dir = os.path.dirname(train_file),
    )
    logger.info(f"adding task for column {label_column}")
    processor.add_task(
            task_type="classification",
            name="text_classification",
            text_column_name="text",
            label_list=label_list,
            metric="mymetrics",
            label_column_name=label_column
        )
    logger.info(f"Got processor: {processor}")
    logger.info(f"Creating silo from file {train_file}")
    silo = farm.data_handler.data_silo.DataSilo(
        processor=processor,
        batch_size=batch_size)
    logger.info(f"Got silo {silo}")
    logger.info("Loading language model")
    language_model = LanguageModel.load(lang_model_dir)

    model = AdaptiveModel(
        language_model=language_model,
        prediction_heads=[predictionhead],
        lm_output_types=["per_sequence"],
        embeds_dropout_prob=0.1,
        device=device)

    return model, silo

if __name__ == "__main__":

    from farm_head_coral import CoralOrdinalRegressionHead
    # head = TextClassificationHead(label_list=labels, num_labels=len(labels))
    # head = CoralOrdinalRegressionHead(label_list=labels)
    head = CoralOrdinalRegressionHead(label_list=labels)

    model, silo = init_model(head, label_list=labels, label_column="sexism_orig_max", train_file=train_file)
    processor = silo.processor
    trainset = silo.data["train"]
    logger.info(f"Trainset is {trainset}")
    example = next(iter(trainset))
    for idx, el in enumerate(example):
        logger.info(f"Example element {idx}/{len(example)}: {el}")
    trainloader = silo.get_data_loader("train")
    batch = next(iter(trainloader))
    logger.info(f"Got a batch: {batch}")
    model.connect_heads_with_processor(processor.tasks)
    head = model.prediction_heads[0]
    logger.info(f"Initialized head:\n{head}")

    ## Now that we have the head, we can run various tests on it

    logger.info(f"label_tensor_name is {head.label_tensor_name}")  # {task_name_}label_ids

    t1 = torch.zeros(1, 768)
    out1 = head(t1)
    logger.info(f"Output for 1,768 tensor: {out1}")

    t2 = torch.zeros(2, 768)
    out2 = head(t2)
    logger.info(f"Output for 2,768 tensor: {out2}")

    kwargs = {
        head.label_tensor_name: torch.tensor([1, 0])
    }
    out2d = out2.detach()
    l2lout2 = head.logits_to_loss(out2d, **kwargs)
    logger.info(f"Output for l2l for 2,768 tensor: {l2lout2}")
    l2pout2 = head.logits_to_probs(out2d, True, **kwargs)
    logger.info(f"Output for l2p/True for 2,768 tensor: {l2pout2}")
    l2pout2 = head.logits_to_probs(out2d, False, **kwargs)
    logger.info(f"Output for l2p/Fasle for 2,768 tensor: {l2pout2}")

    l2rout2 = head.logits_to_preds(out2d, **kwargs)
    logger.info(f"Output for l2preds for 2,768 tensor: {l2rout2}")

    ls = head.prepare_labels(**kwargs)
    logger.info(f"Output for prepare_labels: {ls}")

    # for the formatted predictions we need to run inference for from a dict basically, so we
    # reimplement the important parts from Inferencer._inference_without_multiprocessing here
    dicts = [{"text": "das ist ein Text"}]
    indices = list(range(len(dicts)))
    dataset, tensor_names, problematic_ids, baskets = processor.dataset_from_dicts(
        dicts, indices=indices, return_baskets=True)
    logger.info(f"Got baskets: {baskets}")
    samples = [s for b in baskets for s in b.samples]
    logger.info(f"Got samples: {samples}")
    dataloader = NamedDataLoader(dataset=dataset, sampler=SequentialSampler(dataset), batch_size=batch_size,
                                 tensor_names=tensor_names)
    batch = next(iter(dataloader))
    samples4batch = samples[0:2]
    lm = model.language_model
    modelout = lm(**batch)
    # logger.info(f"Output of LM: {modelout}")
    # run the lm output over the head:
    hdout = head(modelout[1]).detach()
    fpred0 = head.formatted_preds(logits=hdout, preds=None, samples=samples4batch, return_class_probs=False)
    logger.info(f"Formatted preds for logits/False: {fpred0}")