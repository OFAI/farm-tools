"""
Modified copy of the original farm.eval module: Our own OurEvaluator class is the same as
Evaluator, except it also passes all the ids to the metric function (can be configured via
the pass_ids=True/False kwarg.)
"""

from tqdm import tqdm
import torch
import numbers
import logging
import numpy as np
from torch.utils.data import DataLoader

from farm.utils import to_numpy
from farm.eval import Evaluator
from farm.utils import MLFlowLogger as MlLogger
from farm.modeling.adaptive_model import AdaptiveModel
from farm.visual.ascii.images import BUSH_SEP
from farm.evaluation.metrics import compute_report_metrics
from farm.evaluation.metrics import (matthews_corrcoef, simple_accuracy, acc_and_f1, pearson_and_spearman,
  ner_f1_score, f1_macro, squad, mean_squared_error, r2_score, top_n_accuracy, text_similarity_metric,
  registered_metrics)
logger = logging.getLogger(__name__)

def compute_metrics(metric, preds, labels, instids=None, probs=None):
    """
    Calculate the named metric values for the list of predictions vs list of labels.

    :param metric: The name of a predefined metric; a function that takes a prediction list and a label
        list and returns a dict from metric names to values, or recursively a list of metrics.
        Predefined metrics are: mcc, acc, acc_f1, pear_spear, seq_f1, f1_macro, squad, mse, r2,
        top_n_accuracy, text_similarity_metric.
    :type metric: Samples are truncated after this many tokens.
    :param preds: list of predictions
    :param labels: list of target labels
    :return: a dictionary mapping metric names to values.
    """
    assert len(preds) == len(labels)
    if metric == "mcc":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif metric == "acc":
        return simple_accuracy(preds, labels)
    elif metric == "acc_f1":
        return acc_and_f1(preds, labels)
    elif metric == "pear_spear":
        return pearson_and_spearman(preds, labels)
    # TODO this metric seems very specific for NER and doesnt work for
    elif metric == "seq_f1":
        return {"seq_f1": ner_f1_score(labels, preds)}
    elif metric == "f1_macro":
        return f1_macro(preds, labels)
    elif metric == "squad":
        return squad(preds, labels)
    elif metric == "mse":
        return {"mse": mean_squared_error(preds, labels)}
    elif metric == "r2":
        return {"r2": r2_score(preds, labels)}
    elif metric == "top_n_accuracy":
        return {"top_n_accuracy": top_n_accuracy(preds, labels)}
    elif metric == "text_similarity_metric":
        return text_similarity_metric(preds, labels)
    # elif metric == "masked_accuracy":
    #     return simple_accuracy(preds, labels, ignore=-1)
    elif isinstance(metric, list):
        ret = {}
        for m in metric:
            ret.update(compute_metrics(m, preds, labels))
        return ret
    elif metric in registered_metrics:
        metric_func = registered_metrics[metric]
        return metric_func(preds, labels, instids=instids, probs=probs)
    else:
        raise KeyError(metric)



class OurEvaluator(Evaluator):
    """Handles evaluation of a given model over a specified dataset."""

    def __init__(
        self, data_loader, tasks, device, report=True, pass_instids=True, pass_probs=True,
        outdir=None
    ):
        """
        :param data_loader: The PyTorch DataLoader that will return batches of data from the evaluation dataset
        :type data_loader: DataLoader
        :param label_maps:
        :param device: The device on which the tensors should be processed. Choose from "cpu" and "cuda".
        :param metrics: The list of metrics which need to be computed, one for each prediction head.
        :param metrics: list
        :param report: Whether an eval report should be generated (e.g. classification report per class).
        :type report: bool
        :param pass_instids: whether or not to also pass the list of example/instance ids to the metric function (True)
        :type pass_instids: bool
        :param pass_probs: whether or not to also pass the list of probabilities to the metric function (True)
        :type pass_probs: bool
        """
        super().__init__(data_loader=data_loader, tasks=tasks, device=device, report=report)
        self.pass_instids = pass_instids
        self.pass_probs = pass_probs
        self.outdir = outdir

    def eval(self, model, return_preds_and_labels=False, calibrate_conf_scores=False, foldnr=0, global_step=None):
        """
        Performs evaluation on a given model.

        :param model: The model on which to perform evaluation
        :type model: AdaptiveModel
        :param return_preds_and_labels: Whether to add preds and labels in the returned dicts of the
        :type return_preds_and_labels: bool
        :param calibrate_conf_scores: Whether to calibrate the temperature for temperature scaling of the confidence scores
        :type calibrate_conf_scores: bool
        :param foldnr: the fold/repeat number for this evaluation (used for the prediction file name)
        :type foldnr: int
        :param global_step: !!! Unused, but would get the last global stop from the trainer, when evaluating
            on the dev set. This would get passed to the loss aggregation function if there is one, because
            that function could need the global step. FOR NOW UNUSED!
        :return all_results: A list of dictionaries, one for each prediction head. Each dictionary contains the metrics
                             and reports generated during evaluation.
        :rtype all_results: list of dicts
        """
        model.eval()

        # init empty lists per prediction head
        loss_all = [0 for _ in model.prediction_heads]
        aggregated_loss_all = [0 for _ in model.prediction_heads]  # store in all heads the same aggregated loss
        preds_all = [[] for _ in model.prediction_heads]
        label_all = [[] for _ in model.prediction_heads]
        ids_all = [[] for _ in model.prediction_heads]
        probs_all = [[] for _ in model.prediction_heads]
        passage_start_t_all = [[] for _ in model.prediction_heads]
        logits_all = [[] for _ in model.prediction_heads]
        instids_all = []

        for step, batch in enumerate(
            tqdm(self.data_loader, desc="Evaluating", mininterval=10)
        ):
            batch = {key: batch[key].to(self.device) for key in batch}

            with torch.no_grad():

                logits = model.forward(**batch)
                # NOTE: we get here:
                # "The per sample per prediciton head loss whose first two dimensions have length n_pred_heads, batch_size"
                # This is also what gets passed to the loss aggregation function directly in the Adaptive model!
                losses_per_head = model.logits_to_loss_per_head(logits=logits, **batch)
                if model.loss_aggregation_fn is not None:
                    aggregated_loss = np.sum(to_numpy(model.loss_aggregation_fn(losses_per_head, global_step=global_step, batch=batch)))
                else:
                    # this is probably never None if the Adaptive model stores a default method, but in case it is,
                    # we use the same default aggregation: just sum over everythin!
                    aggregated_loss = np.sum(to_numpy(losses_per_head))
                preds = model.logits_to_preds(logits=logits, **batch)
                labels = model.prepare_labels(**batch)
                # adaptive model does not have logits_to_probs, only the textclassification head has this!
                # probs = model.logits_to_probs(logits=logits, **batch)
                # instead, collect probs from all heads
                head_num = 0
                for head, logits_for_head in zip(model.prediction_heads, logits):
                    tmpprobs = head.logits_to_probs(logits=logits_for_head, return_class_probs=False)
                    probs_all[head_num] += list(to_numpy(tmpprobs))
                    head_num += 1


            if self.pass_instids:
                instids_all += list(to_numpy(batch.get("instid")))
            # stack results of all batches per prediction head
            for head_num, head in enumerate(model.prediction_heads):
                loss_all[head_num] += np.sum(to_numpy(losses_per_head[head_num]))
                aggregated_loss_all[head_num] += aggregated_loss
                preds_all[head_num] += list(to_numpy(preds[head_num]))
                label_all[head_num] += list(to_numpy(labels[head_num]))
                # probs_all[head_num] += list(to_numpy(probs[head_num]))
                if head.model_type == "span_classification":
                    ids_all[head_num] += list(to_numpy(batch["id"]))
                    passage_start_t_all[head_num] += list(to_numpy(batch["passage_start_t"]))
                    if calibrate_conf_scores:
                        logits_all[head_num] += list(to_numpy(logits))

        # Evaluate per prediction head
        all_results = []
        for head_num, head in enumerate(model.prediction_heads):
            if head.model_type == "multilabel_text_classification":
                # converting from string preds back to multi-hot encoding
                from sklearn.preprocessing import MultiLabelBinarizer
                mlb = MultiLabelBinarizer(classes=head.label_list)
                # TODO check why .fit() should be called on predictions, rather than on labels
                preds_all[head_num] = mlb.fit_transform(preds_all[head_num])
                label_all[head_num] = mlb.transform(label_all[head_num])
            if head.model_type == "span_classification" and calibrate_conf_scores:
                temperature_previous = head.temperature_for_confidence.item()
                logger.info(f"temperature used for confidence scores before calibration: {temperature_previous}")
                head.calibrate_conf(logits_all[head_num], label_all[head_num])
                temperature_current = head.temperature_for_confidence.item()
                logger.info(f"temperature used for confidence scores after calibration: {temperature_current}")
                temperature_change = (abs(temperature_current - temperature_previous) / temperature_previous) * 100.0
                if temperature_change > 50:
                    logger.warning(f"temperature used for calibration of confidence scores changed by more than {temperature_change} percent")
            if hasattr(head, 'aggregate_preds'):
                # Needed to convert NQ ids from np arrays to strings
                ids_all_str = [x.astype(str) for x in ids_all[head_num]]
                ids_all_list = [list(x) for x in ids_all_str]
                head_ids = ["-".join(x) for x in ids_all_list]
                preds_all[head_num], label_all[head_num] = head.aggregate_preds(preds=preds_all[head_num],
                                                                                labels=label_all[head_num],
                                                                                passage_start_t=passage_start_t_all[head_num],
                                                                                ids=head_ids)


            result = {"loss": loss_all[head_num] / len(self.data_loader.dataset),
                      "aggregated_loss": aggregated_loss_all[head_num] / len(self.data_loader.dataset),
                      "task_name": head.task_name}
            thekwargs = dict(
                metric=head.metric,
                preds=preds_all[head_num],
                labels=label_all[head_num],
            )
            if self.pass_instids:
                thekwargs["instids"] = instids_all
            if self.pass_probs:
                thekwargs["probs"] = probs_all[head_num]
            result.update(compute_metrics(**thekwargs))

            if self.outdir is not None:
                # for now, always require instids and probs here!
                hdprobs = probs_all[head_num]
                hdpreds = preds_all[head_num]
                hdlabel = label_all[head_num]
                assert instids_all is not None
                assert hdprobs is not None
                assert len(hdpreds) == len(hdlabel)
                assert len(instids_all) == len(hdpreds)
                assert len(hdprobs) == len(hdpreds)
                with open(str(self.outdir) + f"/predictions-hd{head_num}-fold{foldnr:02d}.tsv",
                          "wt", encoding="utf-8") as outfp:
                    print("headnr", "foldnr", "instid", "label", "prediction", "probability", sep="\t", file=outfp)
                    for instid, label, pred, prob in zip(instids_all, hdlabel, hdpreds, hdprobs):
                        print(head_num, foldnr, instid, label, pred, prob, sep="\t", file=outfp)

            # Select type of report depending on prediction head output type
            if self.report:
                try:
                    result["report"] = compute_report_metrics(head, preds_all[head_num], label_all[head_num])
                except:
                    logger.error(f"Couldn't create eval report for head {head_num} with following preds and labels:"
                                 f"\n Preds: {preds_all[head_num]} \n Labels: {label_all[head_num]}")
                    result["report"] = "Error"

            if return_preds_and_labels:
                result["preds"] = preds_all[head_num]
                result["labels"] = label_all[head_num]

            all_results.append(result)

        return all_results

    @staticmethod
    def log_results(results, dataset_name, steps, logging=True, print=True, num_fold=None):
        # Print a header
        header = "\n\n"
        header += BUSH_SEP + "\n"
        header += "***************************************************\n"
        if num_fold:
            header += f"***** EVALUATION | FOLD: {num_fold} | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
        else:
            header += f"***** EVALUATION | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
        header += "***************************************************\n"
        header += BUSH_SEP + "\n"
        logger.info(header)

        for head_num, head in enumerate(results):
            logger.info("\n _________ {} _________".format(head['task_name']))
            for metric_name, metric_val in head.items():
                # log with ML framework (e.g. Mlflow)
                if logging:
                    if not metric_name in ["preds","labels"] and not metric_name.startswith("_"):
                        if isinstance(metric_val, numbers.Number):
                            MlLogger.log_metrics(
                                metrics={
                                    f"{dataset_name}_{metric_name}_{head['task_name']}": metric_val
                                },
                                step=steps,
                            )
                # print via standard python logger
                if print:
                    if metric_name == "report":
                        if isinstance(metric_val, str) and len(metric_val) > 8000:
                            metric_val = metric_val[:7500] + "\n ............................. \n" + metric_val[-500:]
                        logger.info("{}: \n {}".format(metric_name, metric_val))
                    else:
                        if not metric_name in ["preds", "labels"] and not metric_name.startswith("_"):
                            logger.info("{}: {}".format(metric_name, metric_val))
