import flwr as fl
import os
import torch
import ray
import numpy as np
import copy
import random
from collections import OrderedDict
from src.log import Checkpoint
from pathlib import Path
from typing import Dict, List
from config import AttrDict
from src.utils import get_func_from_config
from src.apps.clients import test, epochs_to_batches
from src.data import cycle

def train(
    net,
    trainloader,
    optimizer,
    finetune_batch,
    device: str,
    round: int,
    freeze_bn_buffer = False,
    mu: float = 0,
):
    """Train the network on the training set. Returns average
    accuracy and loss. """
    if finetune_batch == 0:
        return 0, 0

    criterion = torch.nn.CrossEntropyLoss()
    if freeze_bn_buffer:
        net.eval()
    else:
        net.train()
    if mu > 0:
        last_round_model = copy.deepcopy(net)
    avg_acc = 0.0
    avg_loss = 0.0
    total = 0
    trainloader = iter(cycle(trainloader))

    for _ in range(finetune_batch):
        images, labels = next(trainloader)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = net(images)

        if mu > 0:
            # FedProx: compute proximal_term
            proximal_term = 0.0
            for w, w_t in zip(net.parameters(), last_round_model.parameters()):
                proximal_term += (w - w_t).norm(2)

            loss = criterion(output, labels) + (mu / 2) * proximal_term
        else:
            loss = criterion(output, labels)
        loss.backward()

        # apply gradients
        optimizer.step()

        # get statistics
        _, predicted = torch.max(output, 1)
        correct = (predicted == labels).sum()
        avg_acc += correct.item()
        avg_loss += loss.item() * images.shape[0] 
        total += images.shape[0]

    return avg_acc / total, avg_loss / total


class ClassificationClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        ckp: Checkpoint,
        local_epochs: int = 1,
        batch_size: int = 32,
        fedprox_mu: float = 0., # fedprox
        exclude_layers: List[str] = [],
        **kwargs
    ):
        self.cid = cid
        self.ckp = ckp
        self.config = ckp.config

        # determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # instantiate model
        self.net_config = self.config.models.net
        arch_fn = get_func_from_config(self.net_config)
        self.net = arch_fn(device=self.device, **self.net_config.args)

        # instantiate data class
        data_config = self.config.data
        data_class = get_func_from_config(data_config)
        self.data_class = data_class(self.ckp, **data_config.args)
        self.dataloader = self.data_class.get_dataloader
        
        # hyperparameters
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        # for fedprox
        self.fedprox_mu = fedprox_mu 
        # for fedbabu
        self.exclude_layers = exclude_layers

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict(
            {k: torch.from_numpy(np.copy(v)) for k, v in params_dict}
        )
        self.net.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, round_config, num_workers=None):
        # print(f"fit() on client cid={self.cid}")
        round_config = AttrDict(round_config)
        self.set_parameters(parameters)

        rnd = int(round_config.current_round)
        # load data for this client and get trainloader
        if num_workers is None:
            num_workers = len(ray.worker.get_resource_ids()["CPU"])

        trainloader = self.dataloader(
            data_pool='train',
            cid=self.cid,
            partition='train',
            batch_size=int(self.batch_size),
            num_workers=num_workers,
            shuffle=True,
            augment=True,
        )

        # send model to device
        self.net.to(self.device)

        # get optimizer type
        optim_func = get_func_from_config(self.net_config.optimizer)
        fixed_layers = list(map(lambda x: x[1],filter(lambda n: n[0] in self.exclude_layers, self.net.named_parameters())))
        trainable_layers = list(map(lambda x: x[1],filter(lambda n: n[0] not in self.exclude_layers, self.net.named_parameters())))

        optimizer = optim_func(
            [
                {'params': fixed_layers, 'lr': 0.},
                {'params': trainable_layers}
            ],
            lr=float(round_config.lr),
            **self.net_config.optimizer.args,
        )

        # convert epochs to num of finetune_batches
        total_fb = epochs_to_batches(self.local_epochs, len(trainloader.dataset), self.batch_size)

        # train
        acc, loss = train(
                            self.net,
                            trainloader,
                            optimizer=optimizer,
                            finetune_batch=int(total_fb),
                            device=self.device,
                            round=rnd,
                            mu=self.fedprox_mu,
                        )

        # return local model
        return self.get_parameters(), len(trainloader.dataset), {"fed_train_acc": acc, "fed_train_loss": loss}

    def evaluate(self, parameters, round_config, num_workers=None, finetune=True, path=None):
        # Personalized FL. Evaluate on test pool
        # print(f"evaluate() on client cid={self.cid}")
        round_config = AttrDict(round_config)
        self.set_parameters(parameters)
        rnd = int(round_config.current_round)

        freeze_bn_buffer = False
        if hasattr(round_config, 'freeze_bn_buffer'):
            freeze_bn_buffer = round_config.freeze_bn_buffer

        test_freeze_bn_buffer = True
        if hasattr(round_config, 'test_freeze_bn_buffer'):
            test_freeze_bn_buffer = round_config.test_freeze_bn_buffer

        if num_workers is None:
            # get num_workers based on ray assignment
            num_workers = len(ray.worker.get_resource_ids()["CPU"])

        finetune_epochs = round_config.finetune_epochs
        
        # send model to device
        self.net.to(self.device)

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
            finetune_epochs=[0]

        # finetune
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

        testloader = self.dataloader(
            data_pool='test',
            cid=self.cid, 
            partition='test', 
            batch_size=50,
            augment=False, 
            num_workers=num_workers,
            path=path
        )

        # converting finetune epochs to finetune batches
        ft_b = [epochs_to_batches(b, len(trainloader.dataset), self.batch_size) for b in ft_b]

        # get optimizer type
        optim_func = get_func_from_config(self.net_config.optimizer)
        optimizer = optim_func(
            self.net.parameters(),
            lr=float(round_config.lr),
            **self.net_config.optimizer.args,
        )

        metrics = {}
        for finetune_batch, ft_epoch in zip(ft_b, finetune_epochs):
            # train
            _, _ = train(
                                self.net,
                                trainloader,
                                optimizer=optimizer,
                                finetune_batch=finetune_batch,
                                device=self.device,
                                round=rnd,
                                freeze_bn_buffer=freeze_bn_buffer,
                                mu=self.fedprox_mu,
                            )

            # evaluate
            loss, accuracy, _ = test(self.net, testloader, device=self.device, freeze_bn_buffer=test_freeze_bn_buffer)
            metrics[f'test_acc_{ft_epoch}'] = float(accuracy)

        # return statistics
        return float(loss), len(testloader.dataset), {**metrics, "accuracy": float(accuracy)}