import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from typing import List
from collections import OrderedDict, defaultdict
from importlib import import_module
import inspect

import logging
logger = logging.getLogger(__name__)

def torch_clamp(x):
    return torch.clamp(x, min=0, max=1)

def set_weights(model: torch.nn.ModuleList, weights: fl.common.Weights) -> None:
    """Set model weights from a list of NumPy ndarrays."""
    state_dict = OrderedDict(
        {
            k: torch.Tensor(np.atleast_1d(v))
            for k, v in zip(model.state_dict().keys(), weights)
        }
    )
    model.load_state_dict(state_dict, strict=True)

def set_weights_multiple_models(nets: list, parameters: fl.common.Weights) -> None:
    # a list of nets and a list of parameters to load in order
    for net in nets:
        assert len(net.state_dict().keys()) <= len(parameters), f'Insufficient parameters to load {type(net).__name__}.'
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        net.load_state_dict(state_dict, strict=True)
        parameters = parameters[len(net.state_dict().keys()):]

def get_updates(original_weights: fl.common.Weights, updated_weights: fl.common.Weights) -> List[torch.Tensor]:
    # extract the updates given two weights
    return [torch.from_numpy(np.copy(up)) - torch.from_numpy(np.copy(op)) for up, op in zip(updated_weights, original_weights)]

def apply_updates(original_weights: fl.common.Weights, updates: List[torch.Tensor]) -> fl.common.Weights:
    # apply updates to original weights
    return [np.copy(op) + up.cpu().detach().numpy() for up, op in zip(updates, original_weights)]

class Step(torch.autograd.Function):
    def __init__(self):
        super(Step, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return (input > 0.).long().float()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

def clampSTE_max(input, max_limit=1.):
    return ClampSTE.apply(input, max_limit)

class ClampSTE(torch.autograd.Function):
    def __init__(self):
        super(ClampSTE, self).__init__()
        
    @staticmethod
    def forward(ctx, input, max_limit=1.):
        return torch.clamp(input, min=0, max=max_limit)
    
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None

class ReLUSTE(torch.autograd.Function):
    def __init__(self):
        super(ReLUSTE, self).__init__()

    @staticmethod
    def forward(ctx, input):
        return F.relu(input, inplace=True)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class KLDBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(KLDBatchNorm1d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.beta = 1. 
        self.init_mode = False
        self.num_features = num_features

        if self.track_running_stats:
            self.register_buffer('init_running_mean', torch.zeros(num_features))
            self.register_buffer('init_running_var', torch.ones(num_features))
        
    def register_init_parameters(self):
        self.init_running_mean = self.running_mean.detach().clone()
        self.init_running_var = self.running_var.detach().clone()
        self.reset_running_stats()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0])
            # use biased var in train
            var = input.var([0], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var

            mean = (1. - self.beta) * self.init_running_mean + self.beta * mean
            var = (1. - self.beta) * self.init_running_var + self.beta * var  

        else:
            if self.init_mode:
                mean = self.init_running_mean
                var = self.init_running_var
            else:
                mean = (1. - self.beta) * self.init_running_mean + self.beta * self.running_mean
                var = (1. - self.beta) * self.init_running_var + self.beta * self.running_var  

        input = (input - mean[None, :]) / (torch.sqrt(var[None, :] + self.eps))
        if self.affine:
            input = input * self.weight[None, :] + self.bias[None, :]

        return input


class KLDBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(KLDBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)

        self.beta = 1. 
        self.init_mode = False
        self.num_features = num_features

        if self.track_running_stats:
            self.register_buffer('init_running_mean', torch.zeros(num_features))
            self.register_buffer('init_running_var', torch.ones(num_features))
        
    def register_init_parameters(self):
        self.init_running_mean = self.running_mean.detach().clone()
        self.init_running_var = self.running_var.detach().clone()
        self.reset_running_stats()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def forward(self, input):
        self._check_input_dim(input)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            mean = input.mean([0, 2, 3])
            # use biased var in train
            var = input.var([0, 2, 3], unbiased=False)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var

            mean = (1. - self.beta) * self.init_running_mean + self.beta * mean
            var = (1. - self.beta) * self.init_running_var + self.beta * var  

        else:
            if self.init_mode:
                mean = self.init_running_mean
                var = self.init_running_var
            else:
                mean = (1. - self.beta) * self.init_running_mean + self.beta * self.running_mean
                var = (1. - self.beta) * self.init_running_var + self.beta * self.running_var  

        input = (input - mean[None, :, None, None]) / (torch.sqrt(var[None, :, None, None] + self.eps))
        if self.affine:
            input = input * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return input

def set_kld_beta(model, betas):
    i = 0 
    for m in model.modules():
        if isinstance(m,(KLDBatchNorm1d,KLDBatchNorm2d)):
            if type(betas) in [int, float]:
                m.beta = betas
            elif torch.is_tensor(betas) and betas.size().numel() == 1:
                m.beta = betas.squeeze()
            else:
                m.beta = betas.squeeze()[i]
                i += 1

def update_mean_and_var(m_x, v_x, N, m_y, v_y, M):
    if M == 1:
        var = v_x
    else:
        var1 = ((N - 1) * v_x + (M - 1) * v_y) / (N + M - 1)
        var2 = (N * M * ((m_x - m_y) ** 2)) / ((N+M)*(N+M-1))
        var = var1 + var2
    mean = (N*m_x + M*m_y) / (N+M)

    return mean, var, N+M

def precompute_kld(model, dataloader, jit_augment, device, eps=1e-5):
    set_kld_bn_mode(model, True) # Use initial model for KLDBatchNorm
    model.eval()

    feat_stats = {}
    def set_hook(name):
        if name not in feat_stats:
            feat_stats[name] = defaultdict(float)

        def hook(m, inp, outp):
            inp = inp[0]
            if len(inp.size()) == 2:
                mean = inp.mean([0])
                var = inp.var([0], unbiased=True)
            else:
                mean = inp.mean([0, 2, 3])
                var = inp.var([0, 2, 3], unbiased=True)
            n = inp.numel() / inp.size(1) 

            with torch.no_grad():
                feat_stats[name]['running_mean'], \
                feat_stats[name]['running_var'], \
                feat_stats[name]['total_size'] = update_mean_and_var(feat_stats[name]['running_mean'],
                                        feat_stats[name]['running_var'],
                                        feat_stats[name]['total_size'],
                                        mean,
                                        var,
                                        n)

        return hook

    hooks = {}
    i = 0
    for m in model.modules():
        if isinstance(m, (KLDBatchNorm1d, KLDBatchNorm2d, nn.BatchNorm2d)):
            hooks[i] = m.register_forward_hook(set_hook(i))
            i += 1

    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            if jit_augment is not None:
                img = jit_augment(img)
            model(img)
    
    for h in hooks.values():
        h.remove()

    set_kld_bn_mode(model, False)

    klds = None

    i = 0
    for m in model.modules():
        if isinstance(m, (KLDBatchNorm1d, KLDBatchNorm2d, nn.BatchNorm2d)):
            if isinstance(m, (KLDBatchNorm1d, KLDBatchNorm2d)):
                init_dist = Normal(m.init_running_mean, torch.sqrt(m.init_running_var + eps))
            else:
                init_dist = Normal(m.running_mean, torch.sqrt(m.running_var + eps))
            client_dist = Normal(feat_stats[i]['running_mean'], torch.sqrt(feat_stats[i]['running_var'] + eps))
            # client_dist = Normal(m.running_mean, m.running_var)

            kl = 0.5 * kl_divergence(init_dist, client_dist) + 0.5 * kl_divergence(client_dist, init_dist)
            m_kl = torch.mean(kl).view(1,1)
            if klds == None:
                klds = m_kl
            else:
                klds = torch.cat([m_kl, klds], dim=1) 
            i += 1

    return klds

def precompute_feat_stats(model, learnable_modules, dataloader, jit_augment, device, mode, no_of_samples=None, eps=1e-5):
    set_kld_bn_mode(model, True)
    model.eval()

    global_feats = mode in ['layerwise', 'layerwise_samples']
    hook_all_layers = mode in ['layerwise', 'layerwise_samples', 'layerwise_local']
    global_model = mode in ['layerwise', 'layerwise_samples', 'layerwise_last']


    feat_stats = {}
    def set_hook(name):
        if name not in feat_stats:
            feat_stats[name] = defaultdict(float)

        def hook(m, inp, outp):
            inp = inp[0]

            if global_feats:
                mean = inp.mean()
                var = inp.var(unbiased=True)
                n = inp.numel()

            else:
                if len(inp.size()) == 2:
                    mean = inp.mean([0])
                    var = inp.var([0], unbiased=True)
                else:
                    mean = inp.mean([0, 2, 3])
                    var = inp.var([0, 2, 3], unbiased=True)
                n = inp.numel() / inp.size(1)

            with torch.no_grad():
                feat_stats[name]['running_mean'], \
                feat_stats[name]['running_var'], \
                feat_stats[name]['total_size'] = update_mean_and_var(feat_stats[name]['running_mean'],
                                        feat_stats[name]['running_var'],
                                        feat_stats[name]['total_size'],
                                        mean,
                                        var,
                                        n)

        return hook

    hooks = {}
    if hook_all_layers:
        i = 0
        for m in model.modules():
            if isinstance(m, learnable_modules):
                hooks[i] = m.register_forward_hook(set_hook(i))
                i += 1
    else: # 'layerwise_last'
        hooks[0] = model.net.fc.register_forward_hook(set_hook(0))
       
    with torch.no_grad():
        for img, _ in dataloader:
            img = img.to(device)
            if jit_augment is not None:
                img = jit_augment(img)
            model(img)
    
    for h in hooks.values():
        h.remove()

    set_kld_bn_mode(model, False)
    # process feat_stats

    if global_model: 
        mask_net_input = None
        for stats in feat_stats.values():
            # if mode in ['layerwise', 'layerwise_samples']:
                # l_stats = torch.cat([torch.mean(stats['running_mean']).view(1), torch.mean(stats['running_var']).view(1)])
            # elif mode == 'layerwise_last':
            #     l_stats = torch.cat([stats['running_mean'], stats['running_var']])

            if global_feats:
                l_stats = torch.cat([stats['running_mean'].view(1), torch.sqrt(stats['running_var'].view(1) + eps)])
            else:
                l_stats = torch.cat([stats['running_mean'], stats['running_var']])

            if mask_net_input is None:
                mask_net_input = l_stats
            else:
                mask_net_input = torch.cat([mask_net_input, l_stats])
        
        if mode == 'layerwise_samples':
            assert no_of_samples is not None
            mask_net_input = torch.cat([torch.log(torch.tensor([no_of_samples])).to(device), mask_net_input])
                
        return mask_net_input.view(1,-1)
    
    else: # layerwise_local for local models
        mask_net_input = {}
        for idx, stats in feat_stats.items():
            mask_net_input[idx] = torch.cat([stats['running_mean'], torch.sqrt(stats['running_var'] + eps)]).view(1,-1)
        return mask_net_input

def set_kld_bn_mode(m, v):
    for n, ch in m.named_children():
        if type(ch) in [KLDBatchNorm1d, KLDBatchNorm2d]:
            ch.init_mode = v
        set_kld_bn_mode(ch, v)

def copy_pretrain_to_kld(pretrain_weights, model):
    load_net_sd = OrderedDict({})
    model_state_dict = model.state_dict()
    if dict in inspect.getmro(type(pretrain_weights)):
        if 'classifier.bias' in pretrain_weights:
            # baseline model
            for k, v in pretrain_weights.items():
                if 'classifier' in k:
                    assert v.shape == model_state_dict[k.replace('classifier','net.fc')].shape
                    load_net_sd[k.replace('classifier','net.fc')] = v
                else:
                    load_net_sd[k.replace('base','net')] = v
        else:
            # pytorch's pretrained imagenet model
            for k, v in pretrain_weights.items():
                if f'net.{k}' in model_state_dict and model_state_dict[f'net.{k}'].size() == v.size():
                    load_net_sd[f'net.{k}'] = v
    else:
        net_keys_wo_ada = [k for k in model_state_dict.keys() if 'init_running' not in k]
        assert len(net_keys_wo_ada) == len(pretrain_weights)

        for k, w in zip(net_keys_wo_ada, pretrain_weights):
            load_net_sd[k] = torch.Tensor(np.atleast_1d(w))

    model.load_state_dict(load_net_sd, strict=False)
    reinit_bn(model)


def reinit_bn(m):
    for n, ch in m.named_children():
        if type(ch) in [KLDBatchNorm1d, KLDBatchNorm2d]:
            ch.register_init_parameters()
        reinit_bn(ch)

