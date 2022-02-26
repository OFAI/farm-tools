"""
Module that contains pre-defined instances of FarmTasks subclasses
"""
import importlib
import math
from abc import ABC, abstractmethod
import copy
import numpy as np
import mlflow

import torch.nn
from farm_processor import OurTextClassificationProcessor
# from farm.data_handler.processor import TextClassificationProcessor as OurTextClassificationProcessor
from farm.evaluation.metrics import simple_accuracy, register_metrics
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import matthews_corrcoef, f1_score, mean_squared_error, mean_absolute_error
from gatenlp.utils import init_logger
from farm_head_coral import CoralOrdinalRegressionHead
from farm_class_head import OurTextClassificationHead
from farm_utils import str2bool, add_cfg
logger = init_logger()


def calculate_class_weights(datasilo, task_name, source="train"):
    """
    Our own implementation since currently the xval datasilo does not provide it!
    :param datasilo: silo that contains the train dataset
    :param task_name: used to get the labels and data
    :param source: which dataset to use, default is "train"
    :return: class weights
    """
    tensor_name = datasilo.processor.tasks[task_name]["label_tensor_name"]
    label_list = datasilo.processor.tasks[task_name]["label_list"]
    tensor_idx = list(datasilo.tensor_names).index(tensor_name)
    # we need at least ONE observation for each label to avoid division by zero in compute_class_weights.
    observed_labels = copy.deepcopy(label_list)
    if source == "all":
        datasets = datasilo.data.values()
    elif source == "train":
        datasets = [datasilo.data["train"]]
    else:
        raise Exception("source argument expects one of [\"train\", \"all\"]")
    for dataset in datasets:
        if dataset is not None:
            observed_labels += [label_list[x[tensor_idx].item()] for x in dataset]
    # TODO scale e.g. via logarithm to avoid crazy spikes for rare classes
    logger.info(f"Label list: {label_list}")
    logger.info(f"Observed labels: {observed_labels}")
    class_weights = list(compute_class_weight("balanced", np.asarray(label_list), observed_labels))
    return class_weights


class ClassificationMetrics:
    def __init__(self, label_list=None, instid_threshold=None):
        """
        Create our own metrics calculator.

        :param label_list: if given, calculate per-class metrics
        :param instid_threshold: if not None, the maximum instid to include in the metrics calculation,
            all instances with instids larger than this value are ignored. Only used if instids are
            actually passed into the metric call!
        """
        if label_list is None:
            label_list = []
        self.label_list = label_list
        self.metric_names = ["acc", "f1_macro", "f1_micro", "mcc"]
        for l in label_list:
            self.metric_names.append(f"f1_{l}")
        self.instid_threshold = instid_threshold

    def preferred(self, result4head):
        """
        Return preferred metric value from a for-a-head result.

        :return: the value of the preferred metric
        """
        val = result4head.get(self.preferred_name())
        if val is None:
            logger.error(f"ERROR: preferred name {self.preferred_name()} not found in result4head: {result4head}")
        return val

    def preferred_name(self):
        return "f1_macro"

    def preferred_mode(self):
        return "max"

    def preferred_best(self, lvalues):
        if self.preferred_mode() == "max":
            return max(lvalues)
        elif self.preferred_mode() == "min":
            return max(lvalues)
        else:
            raise Exception("Allowed modes for preferred metric: max, min")

    def __call__(self, preds, labels, instids=None, probs=None):
        if self.instid_threshold is not None and instids is not None:
            newpreds = []
            newlabels = []
            newinstids = []
            if probs is not None:
                newprobs = []
            else:
                newprobs = None
            for idx in range(len(preds)):
                if instid <= self.instid_threshold:
                    newpreds.append(preds[idx])
                    newlabels.append(labels[idx])
                    newinstids.append(instids[idx])
                    if probs is not None:
                        newprobs.append(probs[idx])
            preds = newpreds
            labels = newlabels
            instids = newinstids
            probs = newprobs
        acc = simple_accuracy(preds, labels).get("acc")
        f1macro = f1_score(y_true=labels, y_pred=preds, average="macro")
        f1micro = f1_score(y_true=labels, y_pred=preds, average="micro")
        wf1macro = f1_score(y_true=labels, y_pred=preds, average="weighted")
        numericpreds = [int(x) for x in preds]
        numericlabels = [int(x) for x in labels]
        mcc = matthews_corrcoef(labels, preds)
        m = {
            "acc": acc,
            "f1_macro": f1macro,
            "f1_macro_weighted": wf1macro,
            "f1_micro": f1micro,
        }
        # to calculate macro mae, mse: calculate the metric for each group of pairs grouped by true label
        # to calculate weighted macro mae, mse: same as macro, but average is calculated by p(true label)
        mae_by_label = []
        mse_by_label = []
        rmse_by_label = []
        mas_by_label = []   # mean absolute score: mae but we map labels/preds to 0,5,6,7,8
        n_label = []
        for label in self.label_list:
            # logger.info(f"DEBUG: processing for label {label}")
            label = str(label)
            m[f"f1_{label}"] = f1_score(y_true=labels, y_pred=preds, labels=[label], average="micro")
            m[f"n_label_{label}"] = labels.count(label)
            m[f"n_preds_{label}"] = preds.count(label)
            # logger.info(f"DEBUG: processing for label {label}, labels={labels.count(label)} preds={preds.count(label)}")
            numericlabel = float(label)
            labels4label = []
            preds4label = []
            for l, p in zip(numericlabels, numericpreds):
                if l == numericlabel:
                    labels4label.append(l)
                    preds4label.append(p)
            # logger.info(f"DEBUG: got labels4labels={labels4label}")
            # logger.info(f"DEBUG: got preds4label={preds4label}")
            # the following increases the label number by 4 to simulate the scoring we used
            # for the annotator comparison to make differences between no sexism and sexism bigger
            labels4label4s = [x if x == 0 else x + 4 for x in labels4label]
            preds4label4s = [x if x == 0 else x + 4 for x in preds4label]
            logger.info(f"DEBUG: got labels4labels4s={labels4label4s}")
            logger.info(f"DEBUG: got preds4label4s={preds4label4s}")
            n_label.append(len(labels4label))
            if len(labels4label) == 0:
                mae_by_label.append(math.nan)
                mse_by_label.append(math.nan)
                rmse_by_label.append(math.nan)
                mas_by_label.append(math.nan)
            else:
                mae_by_label.append(mean_absolute_error(y_true=labels4label, y_pred=preds4label))
                mse_by_label.append(mean_squared_error(y_true=labels4label, y_pred=preds4label))
                rmse_by_label.append(mean_squared_error(y_true=labels4label, y_pred=preds4label, squared=False))
                mas_by_label.append(mean_absolute_error(y_true=labels4label4s, y_pred=preds4label4s))
        mae = mean_absolute_error(y_true=numericlabels, y_pred=numericpreds)
        m["mae"] = mae
        m["mae_macro"] = np.average(mae_by_label)
        m["mae_macro_weighted"] = np.average(mae_by_label, weights=n_label)
        mse = mean_squared_error(y_true=numericlabels, y_pred=numericpreds)
        m["mse"] = mse
        m["mse_macro"] = np.average(mse_by_label)
        m["mse_macro_weighted"] = np.average(mse_by_label, weights=n_label)
        rmse = mean_squared_error(y_true=numericlabels, y_pred=numericpreds, squared=False)
        m["rmse"] = rmse
        m["rmse_macro"] = np.average(rmse_by_label)
        m["rmse_macro_weighted"] = np.average(rmse_by_label, weights=n_label)
        numericlabels4s = [x if x == 0 else x + 4 for x in numericlabels]
        numericpreds4s = [x if x == 0 else x + 4 for x in numericpreds]
        mas = mean_absolute_error(y_true=numericlabels4s, y_pred=numericpreds4s)  # our special "score"
        m["mas"] = mas
        m["mas_macro"] = np.average(mas_by_label)
        m["mas_macro_weighted"] = np.average(mas_by_label, weights=n_label)
        m["mcc"] = mcc
        if instids is not None:
            m["instids"] = len(instids)
        return m


class FarmTasks(ABC):

    def __init__(self, cfg=None):
        assert cfg is not None
        self.cfg = add_cfg(cfg, prefix="fts", types={"use_class_weights": str2bool})
        self.use_class_weights = cfg.get("fts_use_class_weights", True)
        self.nheads = 1

    @abstractmethod
    def get_processor(self, **kwargs):
        """Return instantiated processor"""

    @abstractmethod
    def get_heads(self, silo):
        """Return list of heads. Silo required for stuff like class weights calculation"""
        raise Exception("Must get implemented by subclass!")

    @abstractmethod
    def get_es_metric(self):
        """Return the metric to use for early stopping"""
        raise Exception("Must get implemented by subclass!")

    @abstractmethod
    def get_metric(self):
        """Return the metric for evaluation on the dev set"""
        raise Exception("Must get implemented by subclass!")

    def get_loss_aggregation_fn(self, silo=None):
        return None   # will use default


class FTSingleClassification(FarmTasks):
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.label_list = cfg.get("label_list", ["0", "1"])
        self.task_name = cfg.get("task_name", "head0")
        self.label_column_name = cfg.get("label_column", "target")
        self.text_column_name = cfg.get("text_column", "text")
        register_metrics('mymetrics', ClassificationMetrics(label_list=self.label_list))

    def get_processor(self, **kwargs):
        processor = OurTextClassificationProcessor(
            text_column_name=self.text_column_name,
            instid_column_name="id",
            **kwargs)

        processor.add_task(
            task_type="classification",
            name=self.task_name,
            text_column_name=self.text_column_name,
            label_list=self.label_list,
            metric="mymetrics",
            label_column_name=self.label_column_name
        )
        return processor

    def get_heads(self, silo):
        if self.use_class_weights:
            weights = np.array(
                calculate_class_weights(silo, task_name=self.task_name), dtype=np.float32)
            logger.info(f"Using class weights: {weights}")
        else:
            weights = None
            logger.info(f"Not using class weights!")
        mlflow.log_params({"class_weights": weights})
        self.cfg = add_cfg(self.cfg, prefix="hd0")
        layers = self.cfg.get("hd0_layer_dims", [])
        hddim = int(self.cfg.get("hd_dim", 768))
        if layers:
            layers = [int(x) for x in layers.split(",")]
        all_layers = [hddim]
        all_layers.extend(layers)
        all_layers.append(len(self.label_list))
        dropoutrate = float(self.cfg.get("hd0_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd0_nonlinearity", "ReLU")
        head0 = OurTextClassificationHead(
            hd_dim=hddim,
            layer_dims=all_layers,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            class_weights=weights,
            num_labels=len(self.label_list),
            task_name=self.task_name,
        )
        return [head0]

    def get_metric(self):
        return "mymetrics"

    def get_es_metric(self):
        return "mymetrics"


class FTSxBinclassMulticlass(FarmTasks):
    """Dual head: binary classification, multiclass classification"""
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.nheads = 2
        self.label_list0 = ["0", "1"]
        self.label_list1 = ["0", "1", "2", "3", "4"]
        self.task_name0 = "head0_binclass"
        self.task_name1 = "head1_multiclass"
        self.alpha0 = cfg.get("losses_alpha", 0.5)
        self.alpha1 = 1.0 - self.alpha0
        label_column_names = cfg.get("label_column", "sexism_binmax,sexism_origmax").split(",")
        assert len(label_column_names) == 2
        self.label_column_name0 = label_column_names[0]
        self.label_column_name1 = label_column_names[1]
        self.text_column_name = cfg.get("text_column", "text")
        register_metrics('mymetrics0', ClassificationMetrics(label_list=self.label_list0))
        register_metrics('mymetrics1', ClassificationMetrics(label_list=self.label_list1))

    def get_processor(self, **kwargs):
        processor = OurTextClassificationProcessor(
            text_column_name=self.text_column_name,
            instid_column_name="id",
            **kwargs)

        processor.add_task(
            task_type="classification",
            name=self.task_name0,
            text_column_name=self.text_column_name,
            label_list=self.label_list0,
            metric="mymetrics0",
            label_column_name=self.label_column_name0
        )
        processor.add_task(
            task_type="classification",
            name=self.task_name1,
            text_column_name=self.text_column_name,
            label_list=self.label_list1,
            metric="mymetrics1",
            label_column_name=self.label_column_name1
        )
        return processor

    def get_heads(self, silo):
        if self.use_class_weights:
            weights0 = np.array(
                calculate_class_weights(silo, task_name=self.task_name0), dtype=np.float32)
            logger.info(f"Using class weights for head0: {weights0}")
            weights1 = np.array(
                calculate_class_weights(silo, task_name=self.task_name1), dtype=np.float32)
            logger.info(f"Using class weights for head1: {weights1}")
        else:
            weights0 = None
            weights1 = None
            logger.info(f"Not using class weights!")
        mlflow.log_params({"class_weights0": weights0})
        mlflow.log_params({"class_weights1": weights1})

        self.cfg = add_cfg(self.cfg, prefix="hd0")
        self.cfg = add_cfg(self.cfg, prefix="hd1")
        hddim = int(self.cfg.get("hd_dim", 768))
        layers = self.cfg.get("hd0_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
        all_layers = [hddim]
        all_layers.extend(layers)
        all_layers.append(len(self.label_list0))
        dropoutrate = float(self.cfg.get("hd0_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd0_nonlinearity", "ReLU")
        head0 = OurTextClassificationHead(
            layer_dims=all_layers,
            hd_dim=hddim,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            class_weights=weights0,
            num_labels=len(self.label_list0),
            task_name=self.task_name0,
        )

        layers = self.cfg.get("hd1_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
        all_layers = [hddim]
        all_layers.extend(layers)
        all_layers.append(len(self.label_list1))
        dropoutrate = float(self.cfg.get("hd1_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd1_nonlinearity", "ReLU")

        head1 = OurTextClassificationHead(
            layer_dims=all_layers,
            hd_dim=hddim,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            class_weights=weights1,
            num_labels=len(self.label_list1),
            task_name=self.task_name1,
        )
        return [head0, head1]

    def get_loss_aggregation_fn(self, silo=None):
        def loss_per_head_weightedsum(loss_per_head, global_step=None, batch=None):
            """
            Input: loss_per_head (list of tensors), global_step (int), batch (dict)
            Output: aggregated loss (tensor)
            """
            return (self.alpha0 * loss_per_head[0].sum() + self.alpha1 * loss_per_head[1].sum()) / 11.0
        return loss_per_head_weightedsum
        # return None

    def get_metric(self):
        return "mymetrics0"

    def get_es_metric(self):
        return "mymetrics0"


class FTSxCoral(FarmTasks):
    """Single head: Ordinal regression with CORAL head"""
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.label_column_name = cfg.get("label_column", "sexism_orig_max")
        self.text_column_name = cfg.get("text_column", "text")
        self.label_list = ["0", "1", "2", "3", "4"]
        self.task_name = "head0_coral"
        register_metrics('mymetrics', ClassificationMetrics(label_list=self.label_list))

    def get_processor(self, **kwargs):
        processor = OurTextClassificationProcessor(
            text_column_name=self.text_column_name,
            instid_column_name="id",
            **kwargs)

        processor.add_task(
            task_type="classification",
            name=self.task_name,
            text_column_name=self.text_column_name,
            label_list=self.label_list,
            metric="mymetrics",
            label_column_name=self.label_column_name
        )
        return processor

    def get_heads(self, silo):
        self.cfg = add_cfg(self.cfg, prefix="hd0")
        hddim = int(self.cfg.get("hd_dim", 768))
        layers = self.cfg.get("hd0_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
            layers = [hddim] + layers
        else:
            layers = []
        dropoutrate = float(self.cfg.get("hd0_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd0_nonlinearity", "ReLU")
        head0 = CoralOrdinalRegressionHead(
            label_list=self.label_list,
            layer_dims=layers,
            hd_dim=hddim,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            loss_reduction="none",
            task_name=self.task_name,
        )
        return [head0]

    def get_loss_aggregation_fn(self, silo=None):
        return None

    def get_metric(self):
        return "mymetrics"

    def get_es_metric(self):
        return "mymetrics"


class FTSxMulticlass(FarmTasks):
    """Single head: Multiclass classification"""
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.label_list = ["0", "1", "2", "3", "4"]
        self.task_name = cfg.get("task_name", "head0")
        self.label_column_name = cfg.get("label_column", "target")
        self.text_column_name = cfg.get("text_column", "text")
        register_metrics('mymetrics', ClassificationMetrics(label_list=self.label_list))

    def get_processor(self, **kwargs):
        processor = OurTextClassificationProcessor(
            text_column_name=self.text_column_name,
            instid_column_name="id",
            **kwargs)

        processor.add_task(
            task_type="classification",
            name=self.task_name,
            text_column_name=self.text_column_name,
            label_list=self.label_list,
            metric="mymetrics",
            label_column_name=self.label_column_name
        )
        return processor

    def get_heads(self, silo):
        if self.use_class_weights:
            weights = np.array(
                calculate_class_weights(silo, task_name=self.task_name), dtype=np.float32)
            logger.info(f"Using class weights: {weights}")
        else:
            weights = None
            logger.info(f"Not using class weights!")
        mlflow.log_params({"class_weights": weights})
        self.cfg = add_cfg(self.cfg, prefix="hd0")
        hddim = int(self.cfg.get("hd_dim", 768))
        layers = self.cfg.get("hd0_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
        all_layers = [hddim]
        all_layers.extend(layers)
        all_layers.append(len(self.label_list))
        dropoutrate = float(self.cfg.get("hd0_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd0_nonlinearity", "ReLU")
        head0 = OurTextClassificationHead(
            layer_dims=all_layers,
            hd_dim=hddim,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            class_weights=weights,
            num_labels=len(self.label_list),
            task_name=self.task_name,
        )
        return [head0]


    def get_loss_aggregation_fn(self, silo=None):
        return None

    def get_metric(self):
        return "mymetrics"

    def get_es_metric(self):
        return "mymetrics"


class FTSxOrdinal1(FarmTasks):
    """
    Task that uses our own ordinal regression head as hd0
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.label_column_name = cfg.get("label_column", "sexism_orig_max")
        self.text_column_name = cfg.get("text_column", "text")
        self.label_list = ["0", "1", "2", "3", "4"]
        self.task_name = "head0_ordinal1"
        register_metrics('mymetrics', ClassificationMetrics(label_list=self.label_list))

    def get_processor(self, **kwargs):
        processor = OurTextClassificationProcessor(
            text_column_name=self.text_column_name,
            instid_column_name="id",
            **kwargs)

        processor.add_task(
            task_type="classification",
            name=self.task_name,
            text_column_name=self.text_column_name,
            label_list=self.label_list,
            metric="mymetrics",
            label_column_name=self.label_column_name
        )
        return processor

    def get_heads(self, silo):
        self.cfg = add_cfg(self.cfg, prefix="hd0")
        hddim = int(self.cfg.get("hd_dim", 768))
        layers = self.cfg.get("hd0_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
            layers = [hddim] + layers
        else:
            layers = []
        dropoutrate = float(self.cfg.get("hd0_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd0_nonlinearity", "ReLU")
        head0 = OrdinalRegressionHead1(
            label_list=self.label_list,
            layer_dims=layers,
            hd_dim=hddim,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            loss_reduction="none",
            task_name=self.task_name,
        )
        return [head0]

    def get_loss_aggregation_fn(self, silo=None):
        return None

    def get_metric(self):
        return "mymetrics"

    def get_es_metric(self):
        return "mymetrics"


class FTSxOrdinal1a(FarmTasks):
    """
    Task that uses our own ordinal regression head as hd0, calculations in logspace
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.label_column_name = cfg.get("label_column", "sexism_orig_max")
        self.text_column_name = cfg.get("text_column", "text")
        self.label_list = ["0", "1", "2", "3", "4"]
        self.task_name = "head0_ordinal1a"
        register_metrics('mymetrics', ClassificationMetrics(label_list=self.label_list))

    def get_processor(self, **kwargs):
        processor = OurTextClassificationProcessor(
            text_column_name=self.text_column_name,
            instid_column_name="id",
            **kwargs)

        processor.add_task(
            task_type="classification",
            name=self.task_name,
            text_column_name=self.text_column_name,
            label_list=self.label_list,
            metric="mymetrics",
            label_column_name=self.label_column_name
        )
        return processor

    def get_heads(self, silo):
        self.cfg = add_cfg(self.cfg, prefix="hd0")
        hddim = int(self.cfg.get("hd_dim", 768))
        layers = self.cfg.get("hd0_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
            layers = [hddim] + layers
        else:
            layers = []
        dropoutrate = float(self.cfg.get("hd0_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd0_nonlinearity", "ReLU")
        head0 = OrdinalRegressionHead1a(
            label_list=self.label_list,
            hd_dim=hddim,
            layer_dims=layers,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            loss_reduction="none",
            task_name=self.task_name,
        )
        return [head0]

    def get_loss_aggregation_fn(self, silo=None):
        return None

    def get_metric(self):
        return "mymetrics"

    def get_es_metric(self):
        return "mymetrics"


class FTSxOrdinal2(FarmTasks):
    """
    Task that uses our own ordinal regression head as hd0
    """
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.label_column_name = cfg.get("label_column", "sexism_orig_max")
        self.text_column_name = cfg.get("text_column", "text")
        self.label_list = ["0", "1", "2", "3", "4"]
        self.task_name = "head0_ordinal2"
        register_metrics('mymetrics', ClassificationMetrics(label_list=self.label_list))

    def get_processor(self, **kwargs):
        processor = OurTextClassificationProcessor(
            text_column_name=self.text_column_name,
            instid_column_name="id",
            **kwargs)

        processor.add_task(
            task_type="classification",
            name=self.task_name,
            text_column_name=self.text_column_name,
            label_list=self.label_list,
            metric="mymetrics",
            label_column_name=self.label_column_name
        )
        return processor

    def get_heads(self, silo):
        self.cfg = add_cfg(self.cfg, prefix="hd0")
        layers = self.cfg.get("hd0_layer_dims", [])
        hddim = int(self.cfg.get("hd_dim", 768))
        if layers:
            layers = [int(x) for x in layers.split(",")]
            layers = [hddim] + layers
        else:
            layers = []
        dropoutrate = float(self.cfg.get("hd0_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd0_nonlinearity", "ReLU")
        head0 = OrdinalRegressionHead2(
            label_list=self.label_list,
            hd_dim=hddim,
            layer_dims=layers,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            loss_reduction="none",
            task_name=self.task_name,
        )
        return [head0]

    def get_loss_aggregation_fn(self, silo=None):
        return None

    def get_metric(self):
        return "mymetrics"

    def get_es_metric(self):
        return "mymetrics"


class FTSxBinclassOrdinal1(FarmTasks):
    """Dual head: binary classification, OR with Ordinal1"""
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.nheads = 2
        self.label_list0 = ["0", "1"]
        self.label_list1 = ["0", "1", "2", "3", "4"]
        self.task_name0 = "head0_binclass"
        self.task_name1 = "head1_ordinal1"
        label_column_names = cfg.get("label_column", "sexism_binmax,sexism_origmax").split(",")
        self.alpha0 = cfg.get("losses_alpha", 0.5)
        self.alpha1 = 1.0 - self.alpha0
        assert len(label_column_names) == 2
        self.label_column_name0 = label_column_names[0]
        self.label_column_name1 = label_column_names[1]
        self.text_column_name = cfg.get("text_column", "text")
        register_metrics('mymetrics0', ClassificationMetrics(label_list=self.label_list0))
        register_metrics('mymetrics1', ClassificationMetrics(label_list=self.label_list1))

    def get_processor(self, **kwargs):
        processor = OurTextClassificationProcessor(
            text_column_name=self.text_column_name,
            instid_column_name="id",
            **kwargs)

        processor.add_task(
            task_type="classification",
            name=self.task_name0,
            text_column_name=self.text_column_name,
            label_list=self.label_list0,
            metric="mymetrics0",
            label_column_name=self.label_column_name0
        )
        processor.add_task(
            task_type="classification",
            name=self.task_name1,
            text_column_name=self.text_column_name,
            label_list=self.label_list1,
            metric="mymetrics1",
            label_column_name=self.label_column_name1
        )
        return processor

    def get_heads(self, silo):
        if self.use_class_weights:
            weights0 = np.array(
                calculate_class_weights(silo, task_name=self.task_name0), dtype=np.float32)
            logger.info(f"Using class weights for head0: {weights0}")
            weights1 = np.array(
                calculate_class_weights(silo, task_name=self.task_name1), dtype=np.float32)
            logger.info(f"Using class weights for head1: {weights1}")
        else:
            weights0 = None
            weights1 = None
            logger.info(f"Not using class weights!")
        mlflow.log_params({"class_weights0": weights0})
        mlflow.log_params({"class_weights1": weights1})
        hddim = int(self.cfg.get("hd_dim", 768))
        self.cfg = add_cfg(self.cfg, prefix="hd0")
        self.cfg = add_cfg(self.cfg, prefix="hd1")
        layers = self.cfg.get("hd0_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
        all_layers = [hddim]
        all_layers.extend(layers)
        all_layers.append(len(self.label_list0))
        dropoutrate = float(self.cfg.get("hd0_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd0_nonlinearity", "ReLU")
        head0 = OurTextClassificationHead(
            layer_dims=all_layers,
            hd_dim=hddim,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            class_weights=weights0,
            num_labels=len(self.label_list0),
            task_name=self.task_name0,
        )

        layers = self.cfg.get("hd1_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
            layers = [hddim] + layers
        all_layers = layers
        dropoutrate = float(self.cfg.get("hd1_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd1_nonlinearity", "ReLU")

        head1 = OrdinalRegressionHead1(
            layer_dims=all_layers,
            hd_dim=hddim,
            label_list=self.label_list1,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            class_weights=weights1,
            task_name=self.task_name1,
        )
        return [head0, head1]

    def get_loss_aggregation_fn(self, silo=None):
        def loss_per_head_weightedsum(loss_per_head, global_step=None, batch=None):
            """
            Input: loss_per_head (list of tensors), global_step (int), batch (dict)
            Output: aggregated loss (tensor)
            """
            # print(f"!!!!!!!!!!!!!!!!! len={len(loss_per_head)} {loss_per_head[0].shape} {loss_per_head[1].shape}")
            # Seems the head 1 loss is about one order of magnitude smaller so lets multiply it by 10
            return (self.alpha0 * loss_per_head[0].sum() + self.alpha1 * loss_per_head[1].sum())/11.0
        return loss_per_head_weightedsum
        # return None

    def get_metric(self):
        return "mymetrics0"

    def get_es_metric(self):
        return "mymetrics0"


class FTSxBinclassCoral(FarmTasks):
    """Dual head: binary classification, OR with Coral"""
    def __init__(self, cfg=None):
        super().__init__(cfg)
        self.nheads = 2
        self.label_list0 = ["0", "1"]
        self.label_list1 = ["0", "1", "2", "3", "4"]
        self.task_name0 = "head0_binclass"
        self.task_name1 = "head1_coral"
        label_column_names = cfg.get("label_column", "sexism_binmax,sexism_origmax").split(",")
        self.alpha0 = cfg.get("losses_alpha", 0.5)
        self.alpha1 = 1.0 - self.alpha0
        assert len(label_column_names) == 2
        self.label_column_name0 = label_column_names[0]
        self.label_column_name1 = label_column_names[1]
        self.text_column_name = cfg.get("text_column", "text")
        register_metrics('mymetrics0', ClassificationMetrics(label_list=self.label_list0))
        register_metrics('mymetrics1', ClassificationMetrics(label_list=self.label_list1))

    def get_processor(self, **kwargs):
        processor = OurTextClassificationProcessor(
            text_column_name=self.text_column_name,
            instid_column_name="id",
            **kwargs)

        processor.add_task(
            task_type="classification",
            name=self.task_name0,
            text_column_name=self.text_column_name,
            label_list=self.label_list0,
            metric="mymetrics0",
            label_column_name=self.label_column_name0
        )
        processor.add_task(
            task_type="classification",
            name=self.task_name1,
            text_column_name=self.text_column_name,
            label_list=self.label_list1,
            metric="mymetrics1",
            label_column_name=self.label_column_name1
        )
        return processor

    def get_heads(self, silo):
        if self.use_class_weights:
            weights0 = np.array(
                calculate_class_weights(silo, task_name=self.task_name0), dtype=np.float32)
            logger.info(f"Using class weights for head0: {weights0}")
            weights1 = np.array(
                calculate_class_weights(silo, task_name=self.task_name1), dtype=np.float32)
            logger.info(f"Using class weights for head1: {weights1}")
        else:
            weights0 = None
            weights1 = None
            logger.info(f"Not using class weights!")
        mlflow.log_params({"class_weights0": weights0})
        mlflow.log_params({"class_weights1": weights1})
        hddim = int(self.cfg.get("hd_dim", 768))
        self.cfg = add_cfg(self.cfg, prefix="hd0")
        self.cfg = add_cfg(self.cfg, prefix="hd1")
        layers = self.cfg.get("hd0_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
        all_layers = [hddim]
        all_layers.extend(layers)
        all_layers.append(len(self.label_list0))
        dropoutrate = float(self.cfg.get("hd0_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd0_nonlinearity", "ReLU")
        head0 = OurTextClassificationHead(
            layer_dims=all_layers,
            hd_dim=hddim,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            class_weights=weights0,
            num_labels=len(self.label_list0),
            task_name=self.task_name0,
        )

        layers = self.cfg.get("hd1_layer_dims", [])
        if layers:
            layers = [int(x) for x in layers.split(",")]
            layers = [hddim] + layers
        all_layers = layers
        dropoutrate = float(self.cfg.get("hd1_dropoutrate", 0.2))
        nonlinearity = self.cfg.get("hd1_nonlinearity", "ReLU")

        head1 = CoralOrdinalRegressionHead(
            label_list=self.label_list1,
            layer_dims=all_layers,
            hd_dim=hddim,
            nonlinearity=nonlinearity,
            dropoutrate=dropoutrate,
            loss_reduction="none",
            task_name=self.task_name1,
        )

        return [head0, head1]

    def get_loss_aggregation_fn(self, silo=None):
        def loss_per_head_weightedsum(loss_per_head, global_step=None, batch=None):
            """
            Input: loss_per_head (list of tensors), global_step (int), batch (dict)
            Output: aggregated loss (tensor)
            """
            # print(f"!!!!!!!!!!!!!!!!! len={len(loss_per_head)} {loss_per_head[0].shape} {loss_per_head[1].shape}")
            # Seems the head 1 loss is about one order of magnitude smaller so lets multiply it by 10
            return (self.alpha0 * loss_per_head[0].sum() + self.alpha1 * loss_per_head[1].sum())/11.0
        return loss_per_head_weightedsum
        # return None

    def get_metric(self):
        return "mymetrics0"

    def get_es_metric(self):
        return "mymetrics0"



__all__ = [
    "str2bool",
    "FTSingleClassification",
    "FTSxBinclassMulticlass",
    "FTSxCoral",
    "FTSxMulticlass",
    "FTSxOrdinal1",
    "FTSxOrdinal1a",
    "FTSxOrdinal2",
    "FTSxBinclassOrdinal1",
    "FTSxBinclassCoral",
]
