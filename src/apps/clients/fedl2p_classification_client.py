import flwr as fl
import os
import torch
import torch.nn as nn
import ray
import numpy as np
import traceback
from torch.distributions.categorical import Categorical
from flwr.common import parameters_to_weights
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List
from config import AttrDict
from src.data import cycle
from src.utils import get_func_from_config
from src.apps.clients import ClassificationClient, test, epochs_to_batches
from src.models.model_utils import (
    KLDBatchNorm1d,
    KLDBatchNorm2d, 
    copy_pretrain_to_kld, 
    set_kld_beta, 
    precompute_kld,
    precompute_feat_stats,
)
from src.models.weight_net import weights_init

from torch.autograd import grad

from learn2learn.algorithms.base_learner import BaseLearner
from learn2learn.utils import clone_module, clone_parameters, update_module

def maml_lr_update(model, 
                        fixed_lr=None, 
                        masks=None,
                        grads=None,
                        learnable_lrs=None,
                        no_bn_inner_loop=False):
    if grads is not None and fixed_lr is not None:
        if masks is not None:
            if not no_bn_inner_loop:
                assert len(grads) == len(masks)
            else:
                assert len([g for (n,p), g in zip(model.named_parameters(),grads) \
                            if 'bn' not in n and 'downsample.1' not in n]) == len(masks)

            ms = iter(masks)
        if learnable_lrs:
            if not no_bn_inner_loop:
                assert len(grads) == len(fixed_lr)
            else:
                assert len([g for (n,p), g in zip(model.named_parameters(),grads) \
                            if 'bn' not in n and 'downsample.1' not in n]) == len(fixed_lr)

            lrs = iter(fixed_lr)
        
        for (n, p), g in zip(model.named_parameters(), grads):
            if no_bn_inner_loop and ('bn' in n or 'downsample.1' in n):
                p.update = 0.
                continue

            if learnable_lrs:
                lr = next(lrs)
            else:
                lr = fixed_lr
            if masks is not None:
                lr = lr * next(ms)

            p.update = -lr * g
                
    return update_module(model)

class MAMLwLRs(BaseLearner):
    def __init__(self, model, lr, 
            first_order=False,
            no_bn_inner_loop=False):
        super(MAMLwLRs, self).__init__()
        if type(lr) == float:
            self.learnable_lrs = False
        elif type(lr) in [nn.ParameterList, list]:
            self.learnable_lrs = True
        else:
            raise NotImplementedError()
        self.module = model
        self.lr = lr
        self.first_order = first_order
        self.no_bn_inner_loop = no_bn_inner_loop

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def clone(self, first_order=None):
        if first_order is None:
            first_order = self.first_order

        if self.learnable_lrs:
            return MAMLwLRs(clone_module(self.module),
                       lr=clone_parameters(self.lr),
                       first_order=first_order,
                       no_bn_inner_loop=self.no_bn_inner_loop)
        else:
            return MAMLwLRs(clone_module(self.module),
                       lr=self.lr,
                       first_order=first_order,
                       no_bn_inner_loop=self.no_bn_inner_loop)

    def adapt(self, loss, masks=None, first_order=None, track_grads=False, clip_grad=None):
        if first_order is None:
            first_order = self.first_order
        second_order = not first_order

        gradients = grad(loss,
                         self.module.parameters(),
                         retain_graph=second_order,
                         create_graph=second_order)

        if clip_grad is not None and clip_grad > 0:
            for g in gradients:
                torch.clamp_(g, -clip_grad, clip_grad)
        
        if not second_order and not track_grads:
            with torch.no_grad():
                self.module = maml_lr_update(self.module, 
                                fixed_lr=self.lr, 
                                masks=masks,
                                grads=gradients,
                                learnable_lrs=self.learnable_lrs,
                                no_bn_inner_loop=self.no_bn_inner_loop)
            for p in self.module.parameters():
                p.requires_grad=True
        else:
            self.module = maml_lr_update(self.module, 
                        fixed_lr=self.lr, 
                        masks=masks,
                        grads=gradients,
                        learnable_lrs=self.learnable_lrs,
                        no_bn_inner_loop=self.no_bn_inner_loop)


def train(
    net,
    trainloader,
    optimizer,
    inner_loop_lrs,
    finetune_batch,
    device: str,
    masks=None,
):
    """Train the network on the training set. Returns average
    accuracy and loss. """
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    avg_acc = 0.0
    avg_loss = 0.0
    total = 0
    trainloader = iter(cycle(trainloader))

    for finetune_batch_idx in range(finetune_batch):
        if optimizer is not None:
            optimizer.zero_grad()
        else:
            net.zero_grad()

        images, labels = next(trainloader)
        images, labels = images.to(device), labels.to(device)

        output = net(images)
        loss = criterion(output, labels)
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f'[*Warning] Bad Batch in train(). Loss is nan/inf. Skipping finetune batch {finetune_batch_idx}')
            continue

        loss.backward()

        with torch.no_grad():
            if masks is not None:
                for p, m in zip(net.parameters(), masks):
                    p.grad *= m

        # apply gradients
        if optimizer is not None:
            optimizer.step()
        else:
            with torch.no_grad():
                for p, lr in zip(net.parameters(), inner_loop_lrs):
                    p.copy_(p - lr * p.grad)

        # get statistics
        _, predicted = torch.max(output, 1)
        correct = (predicted == labels).sum()
        avg_acc += correct.item()
        avg_loss += loss.item() * images.shape[0]
        total += images.shape[0]

    return avg_acc / total, avg_loss / total

def meta_mask_net_fwd(model, model_input):
    if type(model) == dict:
        # assert len(model) == len(model_input)
        masks = None
        for idx, m in model.items():
            m_i = model_input[idx]

            if masks is None:
                masks = m(m_i).view(-1)
            else:
                masks = torch.cat([masks, m(m_i).view(-1)])
    else:
        masks = model(model_input).squeeze()
        
    return masks 

def meta_train(
    model,
    meta_nets,
    meta_nets_config,
    rnd_train_meta_nets,
    train_dataloader,
    val_dataloader,
    data_class,
    augment_validation,
    optimizers,
    outer_loop_batches, 
    inner_loop_batches,
    device: str,
    clamp_outer_gradients,
    round: int,
    meta_algorithm='IFT',
    hypergrad=None, 
    lr_net_name=None,
    inner_loop_lrs=None,
    learnable_modules=None,
    loss_threshold=100.,
    fixed_beta=None,
):
    criterion = torch.nn.CrossEntropyLoss()
    avg_s_acc = 0.0
    avg_s_loss = 0.0
    avg_q_acc = 0.0
    avg_q_loss = 0.0

    s_total = 0
    q_total = 0
    use_bn_net = 'bn_net' in meta_nets
    use_lr_net = lr_net_name is not None

    trainloader = iter(cycle(train_dataloader)) 
    if val_dataloader is not None:
        valloader = iter(cycle(val_dataloader))    

    if use_bn_net:
        klds = precompute_kld(model, train_dataloader, data_class.jit_augment, device)

    if use_lr_net:
        meta_net_input = precompute_feat_stats(model, 
                            learnable_modules,
                            train_dataloader, 
                            data_class.jit_augment, 
                            device, 
                            meta_nets_config[lr_net_name].mode,
                            no_of_samples=len(train_dataloader.dataset))

    for ol_idx in range(outer_loop_batches):
        for optimizer in optimizers.values():
            optimizer.zero_grad()
        
        learner = model.clone(first_order=True)
        learner.train()

        masks = None # learning rates
        if use_bn_net:
            betas = meta_nets['bn_net'](klds)
            set_kld_beta(learner, betas)
        else:
            set_kld_beta(learner, fixed_beta)

        if use_lr_net:
            masks = meta_mask_net_fwd(meta_nets[lr_net_name], meta_net_input)

        ### inner-loop
        for inner_loop_idx in range(inner_loop_batches):
            clip_grad = None
            s_images, s_labels = next(trainloader)
            s_images, s_labels = s_images.to(device), s_labels.to(device)
            s_images = data_class.jit_augment(s_images)

            output = learner(s_images)
            inner_loss = criterion(output, s_labels)

            ### Debug if loss is too high. this v. rarely happens
            if inner_loss.item() > loss_threshold:
                print(f'[*Inner Loss] Inner_loss over threshold: {inner_loss} at inner-loop step {inner_loop_idx} for batch size {s_images.size(0)}')                
                clip_grad = 0.5 # hotfix to prevent divergence

            if inner_loop_idx == (inner_loop_batches - 1):
                learner.adapt(inner_loss, masks=masks, track_grads=True, clip_grad=clip_grad) # dependency on LRNet
            else:
                learner.adapt(inner_loss, masks=masks, track_grads=False, clip_grad=clip_grad)
        
        ### outer-loop
        learner.eval()
        if val_dataloader is not None:
            q_images, q_labels = next(valloader)
        else:
            q_images, q_labels = next(trainloader)
        q_images, q_labels = q_images.to(device), q_labels.to(device)
        if augment_validation:
            q_images = data_class.jit_augment(q_images)
        else:
            q_images = data_class.jit_normalize(q_images)

        q_output = learner(q_images)
        outer_loss = criterion(q_output, q_labels)

        if torch.isnan(outer_loss).any() or torch.isinf(outer_loss).any():
            print(f"[*Outer Loop] Warning: Bad batch in outer loop. outer loss is nan/inf in outer-loop step {ol_idx}. Skipping Opt")
            continue

        if meta_algorithm == 'FOMAML':            
            outer_loss.backward()
        elif meta_algorithm == 'IFT':
            s_images, s_labels = next(trainloader)
            s_images, s_labels = s_images.to(device), s_labels.to(device)
            s_images = data_class.jit_augment(s_images)

            output = learner(s_images)
            inner_loss = criterion(output, s_labels)

            if torch.isnan(inner_loss).any() or torch.isinf(inner_loss).any():
                print(f"[*IFT] Warning: bad batch. inner loss is nan/inf in outer-loop step {ol_idx}. Skipping Opt")
                continue

            meta_params = []
            dloss_val_dmetaparams = None

            for learn_net_name in rnd_train_meta_nets:
                if learn_net_name == 'bn_net':
                    list_idx = len(meta_params)
                if type(meta_nets[learn_net_name]) == dict:
                    for local_meta_net in meta_nets[learn_net_name].values():
                        meta_params += list(local_meta_net.parameters())
                else:
                    meta_params += [p for p in meta_nets[learn_net_name].parameters()]

            if inner_loop_lrs is not None:
                meta_params += [p for p in inner_loop_lrs]
      
            if 'bn_net' in rnd_train_meta_nets:
                dloss_val_dmetaparams = torch.autograd.grad(
                    outer_loss,
                    [p for p in meta_nets['bn_net'].parameters()],
                    retain_graph=True,
                    allow_unused=True
                )

            learner_params = [p for p in learner.module.parameters()]

            hypergrads = hypergrad.grad(outer_loss, inner_loss, meta_params, learner_params)
            for p, g in zip(meta_params, hypergrads):
                p.grad = g
            
            if dloss_val_dmetaparams is not None:
                for p, g in zip(meta_params[list_idx:], dloss_val_dmetaparams):
                    p.grad = p.grad + g
        else:
            raise NotImplementedError()

        ### Gradient clamping
        if clamp_outer_gradients > 0.:
            for learn_net_name in rnd_train_meta_nets:
                if type(meta_nets[learn_net_name]) == dict:
                    for local_meta_net in meta_nets[learn_net_name].values():
                        for p in local_meta_net.parameters():
                            if p.requires_grad:
                                p.grad.data.clamp_(-clamp_outer_gradients,clamp_outer_gradients) 
                else:
                    for p in meta_nets[learn_net_name].parameters():
                        if p.requires_grad:
                            p.grad.data.clamp_(-clamp_outer_gradients,clamp_outer_gradients) 
            
            if inner_loop_lrs is not None:
                for p in inner_loop_lrs:
                    if p.requires_grad:
                        p.grad.data.clamp_(-clamp_outer_gradients,clamp_outer_gradients)

        for optimizer in optimizers.values():
            optimizer.step()
    
        model.zero_grad() 

        # logging support
        _, predicted = torch.max(output, 1)
        correct = (predicted == s_labels).sum()
        avg_s_acc += correct.item()
        avg_s_loss += inner_loss.item() * s_images.shape[0]
        s_total += s_images.shape[0]

        # logging query
        _, predicted = torch.max(q_output, 1)
        correct = (predicted == q_labels).sum()
        avg_q_acc += correct.item()
        avg_q_loss += outer_loss.item() * q_images.shape[0]
        q_total += q_images.shape[0]
    
    results =  {"fed_query_acc": avg_q_acc / q_total, 
                "fed_query_loss": avg_q_loss / q_total,
                "fed_support_acc": avg_s_acc / s_total, 
                "fed_support_loss": avg_s_loss / s_total,
                }

    # extract beta
    if use_bn_net:
        with torch.no_grad():
            betas = meta_nets['bn_net'](klds)
            mean_beta = torch.mean(betas).item()
            results = {**results, 'mean_beta': mean_beta}

    # extract sparsity from masks
    if use_lr_net:
        with torch.no_grad():
            masks = meta_mask_net_fwd(meta_nets[lr_net_name], meta_net_input)
            sparsity = 1. - (torch.count_nonzero(masks) / torch.numel(masks)).item()
            results = {**results, "model_sparsity": sparsity, 'mean_mask': torch.mean(masks).item()}

        if inner_loop_lrs is not None:
            results = {**results, "mean_inner_loop_lr": np.mean([p.item() for p in inner_loop_lrs])}

    return results

class FedL2PClassificationClient(ClassificationClient):
    def __init__(self, *args, 
        outer_loop_epochs=1, 
        inner_loop_epochs=5,
        meta_algorithm='IFT', 
        augment_validation=0,
        clamp_outer_gradients=0.,
        meta_nets_rounds={'bn_net': 80}, # used to define which meta_net is defined.
        hypergrad=None,
        val_ratio=1/5,
        learnable_lrs=False,
        no_bn_inner_loop=False,
        fixed_beta=None,
         **kwargs):
        super().__init__(*args, **kwargs)
        meta_nets_rounds = {_m: _v for _m, _v in meta_nets_rounds.items() if _v is not None}

        assert len(meta_nets_rounds) > 0, 'please define a meta net to be initialized'
        assert hasattr(self.config.app.args, 'load_model'), 'Please define pretrain model parameters path'
        assert np.max([v for v in meta_nets_rounds.values()]) == self.config.app.run.num_rounds, \
            'num of FL rounds  must match max no. of rounds to train meta models'
        assert 'bn_net' in meta_nets_rounds or fixed_beta is not None, 'BNNet is not defined. Please define fixed_beta'

        if learnable_lrs:
            assert 'lr_net' in meta_nets_rounds

        self.meta_nets_rounds = meta_nets_rounds
        self.learnable_lrs = learnable_lrs
        self.no_bn_inner_loop = no_bn_inner_loop

        if self.no_bn_inner_loop:
            self.learnable_modules = (nn.Conv2d, nn.Linear)
        else:
            self.learnable_modules = (nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d, KLDBatchNorm1d, KLDBatchNorm2d, nn.Linear)

        self.initialize_meta_nets()
        # load any pretrained meta model
        self.load_pretrain_meta_parameters()

        self.outer_loop_epochs = outer_loop_epochs
        self.inner_loop_epochs = inner_loop_epochs
        self.meta_algorithm = meta_algorithm.upper()
        self.val_ratio = val_ratio

        self.hypergrad = None
        self.first_order = True
        if self.meta_algorithm == 'IFT':
            assert hypergrad is not None
            self.hypergrad = get_func_from_config(hypergrad)
            self.hypergrad = self.hypergrad(**hypergrad.args)

        self.augment_validation = augment_validation 
        self.clamp_outer_gradients = clamp_outer_gradients
        self.fixed_beta = fixed_beta

    def initialize_meta_nets(self):
        self.lr_net_name = None
        self.meta_nets = {}
        self.meta_nets_config = {}

        self.inner_loop_lrs = None
        if self.learnable_lrs:
            init_lr = self.ckp.config.app.on_fit.start_inner_lr
            no_of_learnable_layers = 0
            for m in self.net.modules():
                if isinstance(m, self.learnable_modules):
                    no_of_learnable_layers += 1
                    if m.bias is not None:
                        no_of_learnable_layers += 1
            self.no_of_learnable_layers = no_of_learnable_layers
            self.inner_loop_lrs = nn.ParameterList([nn.Parameter(torch.ones(1) * init_lr) for _ in range(self.no_of_learnable_layers)])

        for meta_net_name, meta_net_rounds in self.meta_nets_rounds.items():
            assert meta_net_name in self.config.models, f'{meta_net_name} not defined in config.models'
            meta_net_config = getattr(self.config.models, meta_net_name)
            arch_fn = get_func_from_config(meta_net_config)
            assert meta_net_name in ['bn_net', 'lr_net']

            if meta_net_name == 'bn_net':
                i_size = len([m for m in self.net.modules() if isinstance(m,(KLDBatchNorm1d, KLDBatchNorm2d))])
                if hasattr(meta_net_config, 'mode') and meta_net_config.mode == 'single':
                    meta_net = arch_fn(input_size=i_size, output_size=1, **meta_net_config.args)
                else:
                    # default is layerwise
                    meta_net = arch_fn(input_size=i_size, output_size=i_size, **meta_net_config.args)
                meta_net.apply(lambda b: weights_init(b, meta_net_config.init.gain_init, meta_net_config.init.bias_init))
            else:
                self.lr_net_name = meta_net_name
                assert meta_net_config.mode in ['layerwise', 'layerwise_last', 'layerwise_local', 'layerwise_samples']

                if meta_net_config.mode in ['layerwise', 'layerwise_last', 'layerwise_samples']: ## global net 
                    if meta_net_config.mode in ['layerwise', 'layerwise_samples']:
                        # (1, 2) for each layer. (1, 2*no. of layers) for input to masknet. Output is no of parameters
                        i_size = 0
                        for m in self.net.modules():
                            if isinstance(m,self.learnable_modules):
                                i_size += 1

                    elif meta_net_config.mode == 'layerwise_last':                
                        # Input (1, 2c). Output is no. of parameters.
                        i_size = self.net.net.fc.in_features

                    i_size *= 2 # mean and std of activations

                    if meta_net_config.mode == 'layerwise_samples':
                        # add no. of samples
                        i_size += 1

                    o_size = 0
                    for m in self.net.modules():
                        if isinstance(m,self.learnable_modules):
                            o_size += 1
                            if m.bias is not None:
                                o_size += 1
                    if self.no_bn_inner_loop:
                        assert o_size == len([n for n, p in self.net.named_parameters() if 'bn' not in n and 'downsample.1' not in n])
                    else:
                        assert o_size == len([n for n, p in self.net.named_parameters()])
                    meta_net = arch_fn(input_size=i_size, output_size=o_size, **meta_net_config.args)
                    meta_net.apply(lambda b: weights_init(b, meta_net_config.init.gain_init, meta_net_config.init.bias_init))

                else: #layerwise_local local nets
                    meta_net = {}
                    idx = 0
                    for m in self.net.modules():
                        if isinstance(m, nn.Conv2d):
                            i_size = m.in_channels
                        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, KLDBatchNorm1d, KLDBatchNorm2d)):
                            i_size = m.num_features
                        elif isinstance(m, nn.Linear):
                            i_size = m.in_features
                            
                        if isinstance(m,self.learnable_modules):
                            i_size *= 2
                            o_size = 2 if m.bias is not None else 1
                            meta_net[idx] = arch_fn(input_size=i_size, output_size=o_size, **meta_net_config.args)
                            meta_net[idx].apply(lambda b: weights_init(b, meta_net_config.init.gain_init, meta_net_config.init.bias_init))
                            idx += 1
                    
            self.meta_nets[meta_net_name] = meta_net
            self.meta_nets_config[meta_net_name] = meta_net_config

    def get_parameters(self):
        params = []
        for meta_net in self.meta_nets.values():
            if type(meta_net) == dict:
                for local_meta_net in meta_net.values():
                    params.extend([val.cpu().numpy() for val in local_meta_net.state_dict().values()])
            else:                    
                params.extend([val.cpu().numpy() for val in meta_net.state_dict().values()])
        if self.learnable_lrs:
            params.extend([p.cpu().detach().numpy() for p in self.inner_loop_lrs])
        return params        

    def set_parameters(self, parameters):
        '''
        Setting parameters of the meta nets
        '''
        offset = 0
        for meta_net in self.meta_nets.values():
            if parameters[offset:]: # there are more parameters to load
                if type(meta_net) == dict:
                    for local_meta_net in meta_net.values():
                        params_dict = zip(local_meta_net.state_dict().keys(), parameters[offset:])
                        state_dict = OrderedDict(
                            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
                        )
                        local_meta_net.load_state_dict(state_dict, strict=True)
                        offset += len(local_meta_net.state_dict().keys())
                else:
                    params_dict = zip(meta_net.state_dict().keys(), parameters[offset:])
                    state_dict = OrderedDict(
                        {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
                    )
                    meta_net.load_state_dict(state_dict, strict=True)
                    offset += len(meta_net.state_dict().keys())
            else:
                break
        if len(parameters[offset:]) > 0 and self.learnable_lrs:
            assert len(parameters[offset:]) == self.no_of_learnable_layers 
            self.inner_loop_lrs = nn.ParameterList([nn.Parameter(torch.ones(1) * inn_lr) 
                for inn_lr in parameters[offset:]])

    def load_pretrain_meta_parameters(self):
        if hasattr(self.config.app.args, 'load_meta_model'):
            pretrain_meta_parameters = self.ckp.load_model_from_run_id(self.config.app.args.load_meta_model, 
                                                        save_as='meta_weights.pkl')
            self.set_parameters(parameters_to_weights(pretrain_meta_parameters))

    def load_pretrain_net_parameters(self):
        '''
        Loading of pretrained model
        '''
        if os.path.exists(self.config.app.args.load_model):
            pretrain_parameters = torch.load(self.config.app.args.load_model)
            copy_pretrain_to_kld(pretrain_parameters, self.net)
        else:
            pretrain_parameters = self.ckp.load_model_from_run_id(self.config.app.args.load_model)
            copy_pretrain_to_kld(parameters_to_weights(pretrain_parameters), self.net)

    def setup_meta_nets(self, rnd):
        '''
        Given the current round, return a list of meta_nets to be trained
        '''
        rnd_train_meta_nets = []
        for meta_net_name, max_net_rounds in self.meta_nets_rounds.items():
            meta_net = self.meta_nets[meta_net_name]
            if type(meta_net) == dict:
                for local_meta_net in meta_net.values():
                    local_meta_net.to(self.device)
            else:
                meta_net.to(self.device)
            if rnd <= max_net_rounds:
                rnd_train_meta_nets.append(meta_net_name)
            else:
                if type(meta_net) == dict:
                    for local_meta_net in meta_net.values():
                        for p in local_meta_net.parameters():
                            p.requires_grad = False
                else:
                    for p in meta_net.parameters():
                        p.requires_grad = False
        
        if self.learnable_lrs and 'lr_net' in rnd_train_meta_nets:
            self.inner_loop_lrs.to(self.device)
        elif self.inner_loop_lrs is not None:
            for p in self.inner_loop_lrs:
                p.requires_grad = False

        return rnd_train_meta_nets

    def fit(self, parameters, round_config, num_workers=None):
        # print(f"fit() on client cid={self.cid}. ")
        round_config = AttrDict(round_config)
        self.load_pretrain_net_parameters()
        self.set_parameters(parameters)

        rnd = int(round_config.current_round)

        # load data for this client and get trainloader
        if num_workers is None:
            num_workers = len(ray.worker.get_resource_ids()["CPU"])

        self.net.to(self.device)
        rnd_train_meta_nets = self.setup_meta_nets(rnd)
        
        train_dataloader, val_dataloader = self.dataloader(
            data_pool='train',
            cid=self.cid,
            partition='train',
            batch_size=int(self.batch_size),
            num_workers=num_workers,
            augment='jit',
            shuffle=True,
            val_ratio=self.val_ratio,
            seed=self.config.seed,
        )

        inner_loop_batches = epochs_to_batches(self.inner_loop_epochs, len(train_dataloader.dataset), self.batch_size)
        if val_dataloader is not None:
            outer_loop_batches = epochs_to_batches(self.outer_loop_epochs, len(val_dataloader.dataset), self.batch_size)
        else:
            outer_loop_batches = epochs_to_batches(self.outer_loop_epochs, len(train_dataloader.dataset), self.batch_size)

        if self.learnable_lrs:
            model = MAMLwLRs(self.net, self.inner_loop_lrs, first_order=self.first_order, no_bn_inner_loop=self.no_bn_inner_loop)
        else:
            model = MAMLwLRs(self.net, float(round_config.inner_lr), first_order=self.first_order, no_bn_inner_loop=self.no_bn_inner_loop)

        optimizers = {}
        for learn_net_name in rnd_train_meta_nets:
            meta_net = self.meta_nets[learn_net_name]
        
            if type(meta_net) == dict:
                optimize_parameters = []
                for local_meta_net in meta_net.values():
                    optimize_parameters += list(local_meta_net.parameters())
            else:
                optimize_parameters = meta_net.parameters()
                
            optimizers[learn_net_name] = get_func_from_config(self.meta_nets_config[learn_net_name].optimizer)(
                optimize_parameters,
                lr=float(getattr(round_config, f'{learn_net_name}_lr')),
                **self.meta_nets_config[learn_net_name].optimizer.args
            )
        
        if self.learnable_lrs and 'lr_net' in rnd_train_meta_nets:

            optimizers['learnable_lrs'] = torch.optim.SGD(
                list(self.inner_loop_lrs),
                lr=0.0001,
                momentum=0.
            )

        # train
        results = meta_train(
                            model=model,
                            meta_nets=self.meta_nets,
                            meta_nets_config=self.meta_nets_config,
                            rnd_train_meta_nets=rnd_train_meta_nets,
                            train_dataloader=train_dataloader,
                            val_dataloader=val_dataloader,
                            data_class=self.data_class,
                            augment_validation=self.augment_validation,
                            optimizers=optimizers,
                            outer_loop_batches=outer_loop_batches,
                            inner_loop_batches=inner_loop_batches,
                            device=self.device,
                            clamp_outer_gradients=self.clamp_outer_gradients,
                            round=rnd,
                            meta_algorithm=self.meta_algorithm,
                            hypergrad=self.hypergrad,
                            lr_net_name=self.lr_net_name,
                            inner_loop_lrs=self.inner_loop_lrs,
                            learnable_modules=self.learnable_modules,
                            fixed_beta=self.fixed_beta
                        )
        
        return self.get_parameters(), len(val_dataloader.dataset), results

    def evaluate(self, parameters, round_config, num_workers=None, finetune=True, path=None):
        # print(f"evaluate() on client cid={self.cid}")
        round_config = AttrDict(round_config)
        
        if round_config.standard_learning:
            self.outer_loop_epochs = round_config.budget_train
            self.fit(parameters, round_config.fit_config)
        else:            
            self.set_parameters(parameters)
        self.load_pretrain_net_parameters()

        rnd = int(round_config.current_round)
        full_eval = round_config.full_eval

        if num_workers is None:
            # get num_workers based on ray assignment
            num_workers = len(ray.worker.get_resource_ids()["CPU"])        

        finetune_epochs = round_config.finetune_epochs if full_eval else round_config.eval_epochs

        if finetune:
            if type(finetune_epochs) != list:
                ft_b = [int(finetune_epochs)]
                finetune_epochs = [int(finetune_epochs)]
            else:
                ft_b = finetune_epochs
                if len(ft_b) > 1:
                    ft_b = [ft_b[0]] + [y - x for x, y in zip(ft_b, ft_b[1:])]
        else:
            ft_b = [0]
            finetune_epochs = [0]

        # send model to device
        self.net.to(self.device)
        for meta_net in self.meta_nets.values():
            if type(meta_net) == dict:
                for local_meta_net in meta_net.values():
                    local_meta_net.to(self.device)
            else:
                meta_net.to(self.device)

        testloader = self.dataloader(
            data_pool='test',
            cid=self.cid, 
            partition='test', 
            batch_size=50, 
            num_workers=num_workers,
            augment=False,
            path=path
        )
        trainloader = self.dataloader(
            data_pool='test',
            cid=self.cid,
            partition='train',
            batch_size=int(self.batch_size),
            num_workers=num_workers,
            augment=True,
            shuffle=True,
            path=path,
        )

        ft_b = [epochs_to_batches(b, len(trainloader.dataset), self.batch_size) for b in ft_b]

        # get optimizer type
        optimizer = None
        if not self.learnable_lrs:            
            optim_func = get_func_from_config(self.net_config.optimizer)
            optimizer = optim_func(
                self.net.parameters(),
                lr=float(round_config.lr),
                **self.net_config.optimizer.args,
            )
        else:
            self.inner_loop_lrs.to(self.device)
            for p in self.inner_loop_lrs:
                p.requires_grad = False

        for meta_net in self.meta_nets.values():
            if type(meta_net) == dict:
                for local_meta_net in meta_net.values():
                    for p in local_meta_net.parameters():
                        p.requires_grad = False
            else:
                for p in meta_net.parameters():
                    p.requires_grad = False

        use_bn_net = 'bn_net' in self.meta_nets
        use_lr_net = self.lr_net_name is not None

        masks = None # learning rates

        with torch.no_grad():
            if use_lr_net:
                meta_net_input = precompute_feat_stats(self.net, 
                                    self.learnable_modules, 
                                    trainloader, 
                                    jit_augment=None, 
                                    device=self.device, 
                                    mode=self.meta_nets_config[self.lr_net_name].mode,
                                    no_of_samples=len(trainloader.dataset))

                masks = meta_mask_net_fwd(self.meta_nets[self.lr_net_name], meta_net_input)

            if use_bn_net:
                klds = precompute_kld(self.net, 
                                    trainloader, 
                                    jit_augment=None, 
                                    device=self.device)   
        
                betas = self.meta_nets['bn_net'](klds)
                set_kld_beta(self.net, betas)
            else:
                set_kld_beta(self.net, self.fixed_beta)
                
        metrics = {}
        for finetune_batch, ft_epoch in zip(ft_b, finetune_epochs):
            acc, loss = train(
                net=self.net,
                trainloader=trainloader,
                optimizer=optimizer,
                inner_loop_lrs=self.inner_loop_lrs,
                finetune_batch=finetune_batch,
                device=self.device,
                masks=masks,
            )

            # evaluate
            loss, accuracy, _ = test(self.net, testloader, device=self.device, freeze_bn_buffer=True)
            metrics[f'test_acc_{ft_epoch}:'] = float(accuracy)

        if use_bn_net:
            if len(betas.squeeze().size()):
                for bn_idx, beta in enumerate(betas.squeeze()):
                    metrics[f'layer{bn_idx}_beta'] = beta.item()
            metrics['mean_beta'] = torch.mean(betas).item()

        if masks is not None:
            metrics['model_sparsity'] = 1. - (torch.count_nonzero(masks) / torch.numel(masks)).item()
            metrics['mean_mask'] = torch.mean(masks).item()
        
            if self.learnable_lrs:
                metrics['mean_inner_loop_lr'] = np.mean([lr.item() for lr in self.inner_loop_lrs])
        
        return float(loss), len(testloader.dataset), metrics
