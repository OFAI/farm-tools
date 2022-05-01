"""
Various helper methods for the farm-related modules
"""
import argparse
import torch
from torch import nn

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_cfg(cfg, types=None, prefix=None):
    """
    Return modified cfg with PREFIX_cfg keys added as PREFIX_keyname. NOTE: all values are strings!
    """
    assert prefix is not None
    if types is None:
        types = {}
    ftsparms = cfg.get(f"{prefix}_cfg", [])
    # in case we are getting an unparsed string here (from hsearch!!), split by ";"
    # NOTE: when passing as a cmd line arg, we can have something like "--fos_cfg parm1=val1 parm2=val2"
    # but in hsearch, we specify this as "parm1=val1;parm2=val2"
    if isinstance(ftsparms, str):
        ftsparms = ftsparms.split(";")
    for ftsparm in ftsparms:
        if ";" in ftsparm:
            raise Exception(f"Got single parameter with semicolon {ftsparm}, separate settings with space on command line")
        if "=" not in ftsparm:
            raise Exception(f"Option {prefix}_cfg must contain pname=pvalue values, got {ftsparm}")
        try:
            pname, pval = ftsparm.split("=")
        except:
            raise Exception(f"Cannot interpret {ftsparm} as key=value assignment")
        if pname in types:
            pval = types[pname](pval)
        cfg[f"{prefix}_"+pname] = pval
    return cfg


class OurFeedForwardBlock(nn.Module):
    """
    A feed forward neural network of variable depth and width. This creates a sequential
    network with layers as given in the layer dims list, where the first number is the
    number of inputs and the last number is the number of outputs. Nonlinearities and dropout
    layers are added between the layers but not after the last layer.
    """

    def __init__(self, layer_dims, nonlinearity=torch.nn.ReLU, dropoutrate=None, add2last=False):
        """
        Create a feed-forward network with the given dimensions for input, intermediate and output
        units. The list must have either 2 or more dimensions or be empty. If empty, this block
        does nothing and acts as a noop.
        Otherwise, there will be the specified number of input units, output units and hidden units per
        intermediate layer, with the specified nonlinearity and an optional dropout layer in between
        but not added to the last layer.

        :param layer_dims: a list of at leasy two dimensions (input, [intermediate...], output)
        :param nonlinearity: the nonlinearity to add
        :param dropoutrate: dropoutrate, if not None and > 0.0, add dropout
        :param add2last: if True appends the nonlinearity/dropout also after the last layer, if applicable
        """
        super().__init__()
        if layer_dims is None:
            layer_dims = []
        self.layer_dims = layer_dims
        if len(layer_dims) == 0:
            self.feed_forward = nn.Sequential()
            return
        if len(layer_dims) == 1:
            raise Exception("Layer dims must be empty (NOOP) or have 2 or more numbers of units")
        # If read from config the input will be string
        n_layers = len(layer_dims) - 1
        layers_all = []
        self.output_size = layer_dims[-1]

        for i in range(n_layers):
            size_in = layer_dims[i]
            size_out = layer_dims[i + 1]
            layer = nn.Linear(size_in, size_out)
            layers_all.append(layer)
            # if it is not the last layer (i.e. n_layers-1), insert nonlinearity and dropout
            if add2last or i < n_layers-1:
                layers_all.append(nonlinearity())
                if dropoutrate:
                    layers_all.append(torch.nn.Dropout(p=dropoutrate))
        self.feed_forward = nn.Sequential(*layers_all)

    def forward(self, X):
        logits = self.feed_forward(X)
        return logits

