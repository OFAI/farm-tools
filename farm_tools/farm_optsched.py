"""
Module that contains pre-defined instances of FarmOptSched subclasses to implement configured
optimizers and schedulers.
"""
from abc import ABC, abstractmethod

from farm.utils import MLFlowLogger
from farm_tools.utils import init_logger
from farm_tools.farm_utils import str2bool, add_cfg
from farm.modeling.optimization import _get_optim as farm_get_opt
from farm.modeling.optimization import get_scheduler as farm_get_sched
logger = init_logger()

class FarmOptSched(ABC):

    def __init__(self, model, n_batches, n_epochs, device, learning_rate, grad_acc_steps=1, cfg=None):
        """
        Basically the same as FARM initialize_optimizer.

        :param model: the model
        :param n_batches: the number of batches per epoch
        :param n_epochs: the (maximum) number of epochs. Note that the maximum number of epochs may be
            not the ideal number to put here if it is rarely/ever reached because of early stopping.
            In that case, specify an average/expected number of epochs instead!
        :param device: device to use
        :param learning_rate: the basic learning rate
        :param grad_acc_steps: number of gradient accumulation steps (default 1)
        :param cfg: configuration object
        """
        assert cfg is not None
        self.cfg = add_cfg(cfg, prefix="fos")
        self.model = model
        self.n_batches = n_batches
        self.n_epochs = n_epochs
        self.device = device
        self.learning_rate = learning_rate
        self.grad_acc_steps = grad_acc_steps
        self.opt_options = None
        self.sched_options = None
        self.opt = None
        self.sched = None
        self.num_opt_steps = int(n_batches / grad_acc_steps) * n_epochs

    @abstractmethod
    def get_optsched(self, **kwargs):
        """
        Return model, optimizer and scheduler.

        The model is returned unchaged but included to be compatible with the FARM initialize_optimizer
        function and perhaps we want to modify the model later anyway.
        """


def log_optsched(cfg, opt_options, sched_options):
    # MLFlowLogger.log_params({
    #     "opt_" + k: v for k, v in opt_options.items() if isinstance(v, (int, float, str))
    # })
    # MLFlowLogger.log_params({
    #     "sched_" + k: v for k, v in sched_options.items() if isinstance(v, (int, float, str))
    # })
    MLFlowLogger.log_params({
        k: v for k, v in cfg.items() if k.startswith("fos_")
    })


class FOSDefault(FarmOptSched):
    def __init__(self, model, n_batches, n_epochs, device, learning_rate, grad_acc_steps=1, cfg=None):
        super().__init__(model, n_batches, n_epochs, device, learning_rate,
                         grad_acc_steps=grad_acc_steps, cfg=cfg)

    def get_optsched(self):
        correct_bias = str2bool(self.cfg.get("fos_correct_bias", "False"))
        weight_decay = float(self.cfg.get("fos_weight_decay", "0.01"))
        self.opt_options = {
            "name": "TransformersAdamW",
            "correct_bias": correct_bias,
            "weight_decay": weight_decay,
            "lr": self.learning_rate
        }
        self.opt = farm_get_opt(self.model, self.opt_options)

        warmup_proportion = float(self.cfg.get("fos_warmup_proportion", "0.4"))
        num_training_steps = float(self.cfg.get("fos_num_training_steps", self.num_opt_steps))
        self.sched_options = {
            "name": "CosineWarmupWithRestarts",
            "warmup_proportion": warmup_proportion,
            "num_training_steps": num_training_steps
        }
        self.sched = farm_get_sched(self.opt, self.sched_options)
        log_optsched(self.cfg, self.opt_options, self.sched_options)
        return self.model, self.opt, self.sched


class FOSLinear(FarmOptSched):
    def __init__(self, model, n_batches, n_epochs, device, learning_rate, grad_acc_steps=1, cfg=None):
        super().__init__(model, n_batches, n_epochs, device, learning_rate,
                         grad_acc_steps=grad_acc_steps, cfg=cfg)

    def get_optsched(self):
        correct_bias = str2bool(self.cfg.get("fos_correct_bias", "False"))
        weight_decay = float(self.cfg.get("fos_weight_decay", "0.01"))
        self.opt_options = {
            "name": "TransformersAdamW",
            "correct_bias": correct_bias,
            "weight_decay": weight_decay,
            "lr": self.learning_rate
        }
        num_training_steps = float(self.cfg.get("fos_num_training_steps", self.num_opt_steps))
        num_warmup_steps = 0.1 * num_training_steps
        num_warmup_steps = float(self.cfg.get("fos_num_warmup_steps", num_warmup_steps))
        self.sched_options = {
            "name": "LinearWarmup",
            "num_warmup_steps": num_warmup_steps,
            "num_training_steps": num_training_steps
        }
        self.opt = farm_get_opt(self.model, self.opt_options)
        self.sched = farm_get_sched(self.opt, self.sched_options)
        log_optsched(self.cfg, self.opt_options, self.sched_options)
        return self.model, self.opt, self.sched


class FOSFrozenLM(FarmOptSched):
    def __init__(self, model, n_batches, n_epochs, device, learning_rate, grad_acc_steps=1, cfg=None):
        super().__init__(model, n_batches, n_epochs, device, learning_rate,
                         grad_acc_steps=grad_acc_steps, cfg=cfg)

    def get_optsched(self):
        correct_bias = str2bool(self.cfg.get("fos_correct_bias", "False"))
        weight_decay = float(self.cfg.get("fos_weight_decay", "0.01"))
        # collect all the parameters which are NOT language model parameters
        params = [p[1]
                 for p in self.model.named_parameters()
                 if not p[0].startswith("language_model") and p[1].requires_grad]

        # manually create the Optimizer, do not use the farm_get_opt method which does not support this!
        self.opt_options = {
            "correct_bias": correct_bias,
            "weight_decay": weight_decay,
            "lr": self.learning_rate,
            "params": params
        }
        from transformers.optimization import AdamW as TransformersAdamW
        # from torch.optim import AdamW
        self.opt = TransformersAdamW([self.opt_options])
        num_training_steps = float(self.cfg.get("fos_num_training_steps", self.num_opt_steps))
        num_warmup_steps = 0.1 * num_training_steps
        num_warmup_steps = float(self.cfg.get("fos_num_warmup_steps", num_warmup_steps))
        self.sched_options = {
            "name": "LinearWarmup",
            "num_warmup_steps": num_warmup_steps,
            "num_training_steps": num_training_steps
        }
        self.sched = farm_get_sched(self.opt, self.sched_options)
        log_optsched(self.cfg, self.opt_options, self.sched_options)
        return self.model, self.opt, self.sched


def get_layer_nr_id(name):
    """
    For the given full layer parameter name, e.g.
    language_model.model.encoder.layer.11.attention.output.dense.bias return the
    numeric layer number (11) and the id ("attention.output.dense.bias")
    """
    if name.startswith("language_model.model.encoder.layer."):
        suf = name[35:]
        idx = suf.index(".")
        layer_nr = int(suf[:idx])
        layer_id = suf[idx+1:]
        return layer_nr, layer_id
    else:
        return None, None


class FOSLLRD1(FarmOptSched):
    """
    Layer-wise Learning Rate Decay: exponential decay:
    for layers number 0..11 use llrd ** (11 - layernr) * LR
    Embedding layer parameters are kept fixed, pooling gets full LR
    """
    def __init__(self, model, n_batches, n_epochs, device, learning_rate, grad_acc_steps=1, cfg=None):
        super().__init__(model, n_batches, n_epochs, device, learning_rate,
                         grad_acc_steps=grad_acc_steps, cfg=cfg)

    def get_optsched(self):
        correct_bias = str2bool(self.cfg.get("fos_correct_bias", "False"))
        weight_decay = float(self.cfg.get("fos_weight_decay", "0.01"))
        llrd = float(self.cfg.get("llrd", 0.95))

        params = []    # iterable of dictionaries for each parameter group
        for pname, pval in self.model.named_parameters():
            layer_nr, layer_id = get_layer_nr_id(pname)
            if pval.requires_grad:
                if pname.startswith("language_model.model.embeddings"):
                    # not used for training (frozen)
                    pass
                elif pname.startswith("language_model.model.encoder.layer."):
                    # actual encoder layer, calculate adapted LR
                    thislr = self.learning_rate * (llrd ** layer_nr)
                    params.append({"params": pval,
                                   "correct_bias": correct_bias,
                                   "weight_decay": weight_decay,
                                   "lr": thislr})
                elif pname.startswith("language_model.model.pooler."):
                    # pooler: full LR
                    params.append({"params": pval,
                                   "correct_bias": correct_bias,
                                   "weight_decay": weight_decay,
                                   "lr": self.learning_rate})
                elif pname.startswith("prediction_heads"):
                    # our own prediction head layers, use unchanged settings
                    params.append({"params": pval,
                                   "correct_bias": correct_bias,
                                   "weight_decay": weight_decay,
                                   "lr": self.learning_rate})

        # manually create the Optimizer, do not use the farm_get_opt method which does not support this!
        self.opt_options = params
        from transformers.optimization import AdamW as TransformersAdamW
        # from torch.optim import AdamW
        self.opt = TransformersAdamW(self.opt_options)

        # Scheduler: use cosine
        num_training_steps = float(self.cfg.get("fos_num_training_steps", self.num_opt_steps))
        num_cycles = float(self.cfg.get("fos_cycles", 1))
        self.sched_options = {
            "name": "CosineWarmupWithRestarts",
            "num_cycles": num_cycles,
            "num_training_steps": num_training_steps
        }
        if "fos_num_warmup_steps" in self.cfg:
            num_warmup_steps = float(self.cfg.get("fos_num_warmup_steps"))
            self.sched_options["num_warmup_steps"] = num_warmup_steps
        else:
            warmup_proportion = float(self.cfg.get("fos_warmup_proportion", "0.4"))
            self.sched_options["warmup_proportion"] = warmup_proportion
        self.sched = farm_get_sched(self.opt, self.sched_options)

        # Log it
        log_optsched(self.cfg, self.opt_options, self.sched_options)

        return self.model, self.opt, self.sched




__all__ = [
    "FOSDefault",
    "FOSLinear",
    "FOSFrozenLM",
    "FOSLLRD1",
]
