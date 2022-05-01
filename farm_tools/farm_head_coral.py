
import os
import torch
import importlib
from torch import nn
from transformers import AutoModelForSequenceClassification
from farm.modeling.prediction_head import PredictionHead
from farm_tools.farm_coral.dataset import levels_from_labelbatch
from farm_tools.farm_coral.losses import coral_loss
from farm_tools.utils import init_logger
from farm_tools.farm_utils import OurFeedForwardBlock

logger = init_logger("FARM-CORAL-head")


class CoralLoss:
    def __init__(self, num_labels, reduction="mean"):
        self.reduction = reduction
        self.num_labels = num_labels
        if reduction == "none":
            self.reduction = None

    def __call__(self, logits, target):
        # logger.info(f"logits={logits}")
        # logger.info(f"target={target}")
        levels = levels_from_labelbatch(target, self.num_labels)
        theloss = coral_loss(logits, levels, importance_weights=None, reduction=self.reduction)
        # logger.info(f"Running MyLoss.forward on {logits.shape}/{target.shape}, returning {theloss.shape}")
        return theloss


class CoralOrdinalRegressionHead(PredictionHead):
    def __init__(
        self,
        layer_dims=None,
        hd_dim=768,
        label_list=None,
        loss_reduction="none",
        nonlinearity="ReLU",
        dropoutrate=None,
        task_name="text_classification",
        **kwargs,
    ):
        """
        """
        # TODO: why does the original text classification head a 
        #    label list attribute?
        logger.info(f"Initializing Coral Head: layer_dims={layer_dims}, hd_dim={hd_dim}, label_list={label_list}")
        super().__init__()
        assert isinstance(label_list, list)
        assert len(label_list) > 1
        if layer_dims is None or len(layer_dims) == 0:
            self.coral_weights = nn.Linear(hd_dim, 1, bias=False)
        else:
            self.nonlinearity = nonlinearity
            mod = importlib.import_module("torch.nn")
            nonlin = getattr(mod, nonlinearity)
            self.coral_weights = nn.Sequential(
                OurFeedForwardBlock(layer_dims, dropoutrate=dropoutrate, nonlinearity=nonlin),
                nn.Linear(layer_dims[-1], 1, bias=False)
            )
        self.label_list = label_list
        self.num_labels = len(label_list)
        self.layer_dims = layer_dims
        self.hd_dim = hd_dim
        self.nonlinearity = nonlinearity
        self.dropoutrate = dropoutrate
        self.coral_bias = torch.nn.Parameter(
                torch.arange(self.num_labels - 1, 0, -1).float() / (self.num_labels-1))
        self.ph_output_type = "per_sequence"
        self.model_type = "text_classification"
        self.task_name = task_name #used for connecting with the right output of the processor

        self.loss_fct = CoralLoss(
            num_labels=self.num_labels,
            reduction=loss_reduction,
        )

        if "label_list" in kwargs:
            logger.warning(f"Ignoring label list from kwargs: {kwargs['label_list']}")
            # TODO: maybe check if equal to the one we pass?
            # self.label_list = kwargs["label_list"]

        self.generate_config()
        logger.info(f"Generated config: {self.config}")
        logger.info(f"Created CoralOrdinalRegressionHead, ignored kwargs={kwargs}")
        logger.info(f"Created head:\n{self}")

    @classmethod
    def load(cls, pretrained_model_name_or_path, revision=None, **kwargs):
        """
        Load a prediction head from a saved FARM or transformers model. `pretrained_model_name_or_path`
        can be one of the following:
        a) Local path to a FARM prediction head config (e.g. my-bert/prediction_head_0_config.json)
        b) Local path to a Transformers model (e.g. my-bert)
        c) Name of a public model from https://huggingface.co/models (e.g. distilbert-base-uncased-distilled-squad)


        :param pretrained_model_name_or_path: local path of a saved model or name of a publicly available model.
                                              Exemplary public name:
                                              - deepset/bert-base-german-cased-hatespeech-GermEval18Coarse

                                              See https://huggingface.co/models for full list
        :param revision: The version of model to use from the HuggingFace model hub. Can be tag name, branch name, or commit hash.
        :type revision: str

        """
        logger.info(f"Running HEAD.load for {pretrained_model_name_or_path}")
        if os.path.exists(pretrained_model_name_or_path) \
                and "config.json" in pretrained_model_name_or_path \
                and "prediction_head" in pretrained_model_name_or_path:
            # a) FARM style
            head = super(CoralOrdinalRegressionHead, cls).load(pretrained_model_name_or_path)
        else:
            # b) transformers style
            # load all weights from model
            full_model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path, revision=revision, **kwargs)
            # init empty head
            head = cls(label_list=full_model.label_list)
            # transfer weights for head from full model
            head.feed_forward.feed_forward[0].load_state_dict(full_model.classifier.state_dict())
            # add label list
            head.label_list = list(full_model.config.id2label.values())
            del full_model

        return head

    def forward(self, X):
        logits = self.coral_weights(X) + self.coral_bias
        #logger.info(f"Running forward on {X.shape}, returning {logits.shape}")
        #logger.info(f"forward got logits={logits}")
        return logits

    def logits_to_loss(self, logits, **kwargs):
        # after forward: gets logits as (batchsize, outputs) plus kwargs:
        # input_ids, padding_mask, setment_ids (all batchsize,inputdim size)
        # text_classification_ids (batchsize, 1)
        # returns batchsize losses
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids
        ret = self.loss_fct(logits, label_ids.view(-1))
        # logger.info(f"Running logits_to_loss on {logits.shape}/kwargs={kwargs}, returning {ret.shape}")
        return ret

    def logits_to_probs(self, logits, return_class_probs, **kwargs):
        probs = torch.sigmoid(logits)
        if return_class_probs:
            probs = probs
        else:
            probs = torch.max(probs, dim=1)[0]
        probs = probs.cpu().numpy()
        # logger.info(f"Running logits_to_probs on {logits.shape}/{return_class_probs}/kwargs={kwargs}, returning {probs.shape}")
        return probs

    def logits_to_preds(self, logits, **kwargs):
        # this gets batchsize,1 logits
        logits = logits.cpu().numpy()
        # logger.info(f"LOGITS={logits}")
        probas = torch.sigmoid(torch.tensor(logits))
        # logger.info(f"PROBAS={probas}")
        predict_levels = probas > 0.5
        pred_ids = torch.sum(predict_levels, dim=1)
        # logger.info(f"PRED_IDS={pred_ids}")
        preds = [self.label_list[int(x)] for x in pred_ids]
        # logger.info(f"Running logits_to_preds on {logits.shape}/kwargs={kwargs}, returning {preds}")
        return preds

    def prepare_labels(self, **kwargs):
        label_ids = kwargs.get(self.label_tensor_name)
        label_ids = label_ids.cpu().numpy()
        # This is the standard doc classification case
        try:
            labels = [self.label_list[int(x)] for x in label_ids]
        # This case is triggered in Natural Questions where each example can have multiple labels
        except TypeError:
            labels = [self.label_list[int(x[0])] for x in label_ids]
        # logger.info(f"Running prepare_labels on kwargs={kwargs}, returning {labels}")
        return labels

    def formatted_preds(self, logits=None, preds=None, samples=None, return_class_probs=False, **kwargs):
        """ Like QuestionAnsweringHead.formatted_preds(), this fn can operate on either logits or preds. This
        is needed since at inference, the order of operations is very different depending on whether we are performing
        aggregation or not (compare Inferencer._get_predictions() vs Inferencer._get_predictions_and_aggregate())"""

        assert (logits is not None) or (preds is not None)

        # When this method is used along side a QAHead at inference (e.g. Natural Questions), preds is the input and
        # there is currently no good way of generating probs
        if logits is not None:
            preds = self.logits_to_preds(logits)
            probs = self.logits_to_probs(logits, return_class_probs)
        else:
            probs = [None] * len(preds)

        # TODO this block has to do with the difference in Basket and Sample structure between SQuAD and NQ
        try:
            contexts = [sample.clear_text["text"] for sample in samples]
        # This case covers Natural Questions where the sample is in a QA style
        except KeyError:
            contexts = [sample.clear_text["question_text"] + " | " + sample.clear_text["passage_text"] for sample in samples]

        contexts_b = [sample.clear_text["text_b"] for sample in samples if "text_b" in  sample.clear_text]
        if len(contexts_b) != 0:
            contexts = ["|".join([a, b]) for a,b in zip(contexts, contexts_b)]

        res = {"task": "text_classification", 
               "task_name": self.task_name,
               "predictions": []}
        for pred, prob, context in zip(preds, probs, contexts):
            if not return_class_probs:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": f"{pred}",
                    "probability": prob,
                }
            else:
                pred_dict = {
                    "start": None,
                    "end": None,
                    "context": f"{context}",
                    "label": "class_probabilities",
                    "probability": prob,
                }

            res["predictions"].append(pred_dict)
        return res


