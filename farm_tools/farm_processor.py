import logging

# from farm.data_handler.processor import TextClassificationProcessor
from farm.data_handler.dataset import convert_features_to_dataset
from farm.data_handler.processor import Processor
from farm.data_handler.samples import (
    Sample,
    SampleBasket,
)
from farm.data_handler.utils import read_tsv

logger = logging.getLogger(__name__)


class OurTextClassificationProcessor(Processor):
    """
    Used to handle the text classification datasets that come in tabular format (CSV, TSV, etc.)
    """

    def __init__(
            self,
            tokenizer,
            max_seq_len,
            data_dir,
            label_list=None,
            metric=None,
            train_filename="train.tsv",
            dev_filename=None,
            test_filename="test.tsv",
            dev_split=0.1,
            dev_stratification=False,
            delimiter="\t",
            quote_char="'",
            skiprows=None,
            label_column_name="label",
            multilabel=False,
            header=0,
            proxies=None,
            max_samples=None,
            text_column_name="text",
            instid_column_name=None,
            **kwargs
    ):
        self.instid_column_name = instid_column_name
        # Custom processor attributes
        self.delimiter = delimiter
        self.quote_char = quote_char
        self.skiprows = skiprows
        self.header = header
        self.max_samples = max_samples
        self.dev_stratification = dev_stratification
        logger.warning(f"Currently no support in Processor for returning problematic ids")

        super(OurTextClassificationProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={},
            proxies=proxies,
        )
        if metric and label_list:
            if multilabel:
                task_type = "multilabel_classification"
            else:
                task_type = "classification"
            self.add_task(name="text_classification",
                          metric=metric,
                          label_list=label_list,
                          label_column_name=label_column_name,
                          text_column_name=text_column_name,
                          task_type=task_type)
        else:
            logger.info("Initialized processor without tasks. Supply `metric` and `label_list` to the constructor for "
                        "using the default task or add a custom task later via processor.add_task()")

    def file_to_dicts(self, file: str) -> [dict]:
        column_mapping = {}
        for task in self.tasks.values():
            column_mapping[task["label_column_name"]] = task["label_name"]
            column_mapping[task["text_column_name"]] = "text"
        if self.instid_column_name is not None:
            column_mapping[self.instid_column_name] = "instid"
        dicts = read_tsv(
            filename=file,
            delimiter=self.delimiter,
            skiprows=self.skiprows,
            quotechar=self.quote_char,
            rename_columns=column_mapping,
            header=self.header,
            proxies=self.proxies,
            max_samples=self.max_samples
            )
        if self.instid_column_name is None:
            for idx, d in enumerate(dicts):
                d["instid"] = idx
        return dicts

    def dataset_from_dicts(self, dicts, indices=None, return_baskets=False, debug=False):
        self.baskets = []
        # Tokenize in batches
        texts = [x["text"] for x in dicts]
        tokenized_batch = self.tokenizer.batch_encode_plus(
            texts,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
            return_token_type_ids=True,
            return_attention_mask=True,
            truncation=True,
            max_length=self.max_seq_len,
            padding="max_length"
        )
        input_ids_batch = tokenized_batch["input_ids"]
        segment_ids_batch = tokenized_batch["token_type_ids"]
        padding_masks_batch = tokenized_batch["attention_mask"]
        tokens_batch = [x.tokens for x in tokenized_batch.encodings]

        # From here we operate on a per sample basis
        for dictionary, input_ids, segment_ids, padding_mask, tokens in zip(
                dicts, input_ids_batch, segment_ids_batch, padding_masks_batch, tokens_batch
        ):

            tokenized = {}
            if debug:
                tokenized["tokens"] = tokens

            if self.instid_column_name is None:
                idname = "instid"
            else:
                idname = self.instid_column_name
            feat_dict = {"input_ids": input_ids,
                         "padding_mask": padding_mask,
                         "segment_ids": segment_ids,
                         }

            # Create labels
            # i.e. not inference
            if not return_baskets:
                label_dict = self.convert_labels(dictionary)
                feat_dict.update(label_dict)

            feat_dict["instid"] = int(dictionary.get(idname, 0))

            # Add Basket to self.baskets
            curr_sample = Sample(id=None,
            # curr_sample = Sample(id=dictionary.get(self.instid_column_name),
                                 clear_text=dictionary,
                                 tokenized=tokenized,
                                 features=[feat_dict])
            curr_basket = SampleBasket(id_internal=None,
                                       raw=dictionary,
                                       id_external=None,
                                       samples=[curr_sample])
            self.baskets.append(curr_basket)

        if indices and 0 not in indices:
            pass
        else:
            self._log_samples(1)

        # TODO populate problematic ids
        problematic_ids = set()
        dataset, tensornames = self._create_dataset()
        if return_baskets:
            return dataset, tensornames, problematic_ids, self.baskets
        else:
            return dataset, tensornames, problematic_ids

    def convert_labels(self, dictionary):
        ret = {}
        # Add labels for different tasks
        for task_name, task in self.tasks.items():
            label_name = task["label_name"]
            label_raw = dictionary[label_name]
            label_list = task["label_list"]
            if task["task_type"] == "classification":
                # id of label
                label_ids = [label_list.index(label_raw)]
            elif task["task_type"] == "multilabel_classification":
                # multi-hot-format
                label_ids = [0] * len(label_list)
                for l in label_raw.split(","):
                    if l != "":
                        label_ids[label_list.index(l)] = 1
            ret[task["label_tensor_name"]] = label_ids
        return ret

    def _create_dataset(self):
        # TODO this is the proposed new version to replace the mother function
        features_flat = []
        basket_to_remove = []
        for basket in self.baskets:
            if self._check_sample_features(basket):
                for sample in basket.samples:
                    features_flat.extend(sample.features)
            else:
                # remove the entire basket
                basket_to_remove.append(basket)
        dataset, tensor_names = convert_features_to_dataset(features=features_flat)
        return dataset, tensor_names
