import torch
import numpy as np 

def softmax(x, T=1.0):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x/T) / np.sum(np.exp(x/T), axis=0)

def get_layer(model, layer_name):
    assert layer_name.endswith('.weight') or layer_name.endswith('.bias'), 'layer name must be learnable (end with .weight or .bias'
    layer = model
    for attrib in layer_name.split('.'):
        layer = getattr(layer, attrib)
    return layer