import os, sys
import traceback
import json
import torch
import logging
from six.moves import http_client
from sagemaker_pytorch_serving_container.default_pytorch_inference_handler import DefaultPytorchInferenceHandler
from sagemaker_inference.content_types import ANY as CT_ANY
from sagemaker_inference.content_types import UTF8_TYPES as CT_UTF8_TYPES
from sagemaker_inference.content_types import JSON as CT_JSON
from sagemaker_inference import environment, utils
from sagemaker_inference.errors import BaseInferenceToolkitError, GenericInferenceToolkitError
from sagemaker_inference.transformer import Transformer
from farm.infer import Inferencer
import farm.infer, farm.utils, farm.modeling.prediction_head, farm.data_handler.processor
# Note: the following imports are necessary to register the classes as subclasses for Pytorch,
# only if this is done we can load them!
from farm_class_head import OurTextClassificationHead
from farm_head_coral import CoralOrdinalRegressionHead
from farm_class_head import OurTextClassificationHead
from farm_processor import OurTextClassificationProcessor



ENABLE_MULTI_MODEL = os.getenv("SAGEMAKER_MULTI_MODEL", "false") == "true"

class FarmHandler:

    def init_logger(self):
        # get the root logger
        fmt = "%(asctime)s|%(levelname)s|%(name)s|%(message)s"
        lvl = os.environ.get("FARM_HANDLER_LOGGING_LEVEL", "WARN")
        rl = logging.getLogger()
        rl.setLevel(lvl)
        # NOTE: basicConfig does nothing if there is already a handler, so it only runs once, but we create the additional
        # handler for the file, if needed, only if the root logger has no handlers yet as well
        addhandlers = []
        fmt = logging.Formatter(fmt)
        hndlr = logging.StreamHandler(sys.stderr)
        hndlr.setFormatter(fmt)
        addhandlers.append(hndlr)
        logging.basicConfig(level=lvl, handlers=addhandlers)
        # now get the handler for name
        logger = logging.getLogger("FarmHandler")
        # try to configure farm loggers as well: suppress most of this at all times
        farm.infer.logger.setLevel(lvl)
        farm.utils.logger.setLevel(logging.ERROR)
        farm.modeling.prediction_head.logger.setLevel(logging.ERROR)
        farm.data_handler.processor.logger.setLevel(logging.ERROR)
        return logger

    def __init__(self):
        self.model = None
        self.device = None
        self.context = None
        self.manifest = None
        self.map_location = None
        self.logger = self.init_logger()
        self.logger.debug("FarmHandler init")

    def initialize(self, context):
        self.logger.debug(f"FarmHandler: running initialize")
        properties = context.system_properties
        self.map_location = "cuda" if torch.cuda.is_available() and properties.get("gpu_id") is not None else "cpu"
        self.device = torch.device(
            self.map_location + ":" + str(properties.get("gpu_id"))
            if torch.cuda.is_available() and properties.get("gpu_id") is not None
            else self.map_location
        )
        self.manifest = context.manifest
        self.logger.debug(f"FarmHandler: map_location is {self.map_location}")
        self.logger.debug(f"FarmHandler: device is {self.device}")
        self.logger.debug(f"FarmHandler: manifest is {self.manifest}")
        model_dir = properties.get("model_dir")
        self.model_dir = model_dir
        self.logger.debug(f"FarmHandler: loading model from {model_dir}")
        self.model = self.model_fn(model_dir)
        self.logger.debug(f"FarmHandler: model is {self.model}")
        # TODO: maybe split this into loading the actual torch model as either eager or torchscript plus additional model files here!
        # see https://github.com/pytorch/serve/tree/fd4e3e8b72bed67c1e83141265157eed975fec95/ts/torch_handler
        # TODO: move actual model to device
        # self.model.to(self.device)
        # self.model.eval()
        self.logger.debug("FarmHandler: initialize finished")

    def handle(self, data, context):
        try:
            if self.model is None:
                self.model_fn(self.model_dir)
            self.logger.debug("\n>>>>>>>>>>>>>>>>>>>> data length:", len(data),"\n")
            input_datas_orig = [d.get("body") for d in data]
            self.logger.debug("\n>>>>>>>>>>>>>>>>>>>> request_processor length:", len(context.request_processor),"\n")
            self.logger.debug("\n>>>>>>>>>>>>>>>>>>>> input_datas_orig length:", len(input_datas_orig),"\n")
            input_datas = []
            content_types = []
            response_content_types = []
            for input_data, request_processor in zip(input_datas_orig, context.request_processor):
                self.logger.debug("\n>>>>>>>>>>>>>>>>>>>> request_processor:", request_processor, "\n")
                request_property = request_processor.get_request_properties()
                content_type = utils.retrieve_content_type_header(request_property)
                self.logger.debug("\n>>>>>>>>>>>>>>>>>>>> content type:", content_type, "\n")
                accept = request_property.get("Accept") or request_property.get("accept")
                self.logger.debug("\n>>>>>>>>>>>>>>>>>>>> accept:", accept, "\n")
                if not accept or accept == CT_ANY:
                    accept = CT_JSON
                if content_type in CT_UTF8_TYPES:
                    input_data = input_data.decode("utf-8")
                input_datas.append(input_data)
                content_types.append(content_type)
                response_content_types.append(accept)
                self.logger.debug("\n>>>>>>>>>>>>>>>>>>>> input datas now:", input_datas,"\n")
                self.logger.debug("\n>>>>>>>>>>>>>>>>>>>> content_types now:", content_types,"\n")
                self.logger.debug("\n>>>>>>>>>>>>>>>>>>>> response_content_types now:", response_content_types,"\n")
            data = self.input_fn(input_datas, content_types)
            predictions = self.predict_fn(data)
            responses = self.output_fn(predictions, response_content_types)
            for idx, response_content_type in enumerate(response_content_types):
                context.set_response_content_type(idx, response_content_type)
            return responses
        except Exception as e:  # pylint: disable=broad-except
            trace = traceback.format_exc()
            if isinstance(e, BaseInferenceToolkitError):
                return Transformer.handle_error(context, e, trace)
            else:
                return Transformer.handle_error(
                    context,
                    GenericInferenceToolkitError(http_client.INTERNAL_SERVER_ERROR, str(e)),
                    trace,
                )

    def model_fn(self, model_dir):
        use_gpu = self.device != "cpu"
        inferencer = Inferencer.load(model_dir,
                                     batch_size=32,
                                     gpu=use_gpu,
                                     return_class_probs=False,
                                     disable_tqdm=True,

                                     num_processes=1)
        return inferencer

    def input_fn(self, rbodies, rcontenttypes):
        self.logger.debug(f"!!!!!!!!!!!!!!!! Called input_fn with {rbodies}, {rcontenttypes}")
        # NOT sure yet if we get this already parsed or not here
        data = []
        for rbody, rcontenttype in zip(rbodies, rcontenttypes):
            if isinstance(rbody, str):
                data.append(json.loads(rbody))
            else:
                data.append(rbody)
        return data

    def predict_fn(self, data):
        self.logger.debug(f"!!!!!!!!!!!!!!!!!!!! Called predict_fn with {data} of type {type(data)}")
        ret = self.model.inference_from_dicts(data)
        self.logger.debug(f"#########################!!!!!!!!!!!!!! Got result: \n{ret}\n")
        results = []  # we will store one dict for each data instance, where the dict contains keys for each head
        # ret is a list of heads, for each head we get a list of dicts, in our case only a single dict per head
        # The dict per head contains task_name, task, and predictions
        # predictions is a list of dicts with relevant keys: context, label, probability
        
        # first, create a list of task_names and a list of prediction lists
        task_names = [hd[0]["task_name"] for hd in ret]
        predictionlists = [hd[0]["predictions"] for hd in ret]
        # now create the results: zip all the prediction lists
        for predsperhead in zip(*predictionlists): 
            # create a dict where the keys are prefixed with the corresponding task name for the head
            result = {}
            for task_name, preddict in zip(task_names, predsperhead):
                result["text"] = preddict["context"]
                result["label_"+task_name] = preddict["label"]
                result["probability_"+task_name] = float(preddict["probability"])
            self.logger.debug(f">>>> APPENING RESULT: {result}")
            results.append(result)
        self.logger.debug(f">>>>>>>>>>>>> RETURNING RESULTS: {results}")
        return results

    def output_fn(self, predictions, contenttype):
        self.logger.debug(f"!!!!!!!!!!!!!!!!!! Called output_fn with {len(predictions)} preds: {predictions} of type {type(predictions[0])}, and {len(contenttype)} cts:  {contenttype}")
        return predictions


if __name__ == "__main__":
    # for quick testing of the main functions: loading and classifying some text
    handler = FarmHandler()
    handler.model_dir = "model"
    handler.model = handler.model_fn(handler.model_dir)
    ret = handler.predict_fn([{'text': 'Die tussi weiss gar nichts'}])
    print("GOT ret=", ret)
