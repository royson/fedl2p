import sys
import os
import re
import ast
import yaml
import copy
import numpy as np

import logging
logger = logging.getLogger(__name__)

class PrettySafeLoader(yaml.SafeLoader):
    '''
    Allows yaml to load tuples. Credits to Matt Anderson. See: 
    https://stackoverflow.com/questions/9169025/how-can-i-add-a-python-tuple-to-a-yaml-file-using-pyyaml
    '''
    def construct_python_tuple(self, node):
        return tuple(self.construct_sequence(node))

PrettySafeLoader.add_constructor(
    u'tag:yaml.org,2002:python/tuple',
    PrettySafeLoader.construct_python_tuple)

def read_yaml(filepath):
    with open(filepath, 'r') as stream:
        try:
            return yaml.load(stream, Loader=PrettySafeLoader)
        except yaml.YAMLError as exc:
            logger.error(exc)
            return {}

def get_args():
    args = {}
    basepath = os.path.dirname(__file__)
    args = merge_dict(args, read_yaml(os.path.join(basepath, 'configs', 'default.yaml')))
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.endswith('.yaml'):
                args = merge_dict(args, read_yaml(arg))
            elif len(arg.split('=')) == 2:
                args = merge_dict(args, attr_to_dict(arg))
            else:
                logger.warning(f'unrecognizable argument: {arg}')

    return args

class AttrDict(dict):
    def __init__(self, d={}):
        super(AttrDict, self).__init__()
        for k, v in d.items():
            self.__setitem__(k, v)

    def __setitem__(self, k, v):
        if isinstance(v, dict):
            v = AttrDict(v)
        super(AttrDict, self).__setitem__(k, v)

    def __getattr__(self, k):
        try:
            return self.__getitem__(k)
        except KeyError:
            raise AttributeError(k)

    __setattr__ = __setitem__

def attr_to_dict(attr):
    '''
        Transforms attr string to nested dict
    '''
    nested_k, v = attr.split('=')
    ks = nested_k.split('.')
    d = {}
    ref = d
    while len(ks) > 1:
        k = ks.pop(0)
        ref[k] = {}
        ref = ref[k]

    ref[ks.pop()] = assign_numeric_type(v)

    return d
 
def assign_numeric_type(v):
    if re.match(r'^-?\d+(?:\.\d+)$', v) is not None:
        return float(v)
    elif re.match(r'^-?\d+$', v) is not None:
        return int(v)
    elif re.match(r'^range\(-?\d+,-?\d+,-?\d+\)$', v) is not None:
        r_nos = v.split('range(')[-1][:-1].split(',')
        return list(range(int(r_nos[0]), int(r_nos[1]), int(r_nos[2])))
    elif re.match(r'^[\d_.]+$', v) is not None:
        return str(v)
    elif v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    elif v.lower() == 'null':
        return None
    else: 
        try:
            return ast.literal_eval(v)
        except (SyntaxError, ValueError) as e:
            return v

def merge_dict(a, b):
    '''
        merge nested dictionary b into nested dictionary a
    '''
    assert isinstance(a, dict) and isinstance(b, dict)
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dict(a[key], b[key])
            else:
                a[key] = b[key]
        else:
            a[key] = b[key]
    return a

