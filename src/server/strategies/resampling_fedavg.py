import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

from collections import OrderedDict, deque
from src.models.model_utils import get_updates
from src.server.strategies.utils import softmax
from src.server.strategies import FedAvg

from pprint import pformat

from typing import Dict, Optional, Tuple, List
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
    Parameters,
    Weights,
    Scalar,
    FitRes,
    FitIns,
    parameters_to_weights,
    weights_to_parameters,
)

import logging
logger = logging.getLogger(__name__)

class ResamplingFedAvg(FedAvg):
    '''
        Similar to src.server.strategies.FedAvg.

        Compute weightings based on the gradient 
        of an unbiased validation set on `model_name`.

        `calc_weighting_layers` determine the set of weights and biases that is used to compute the weights.

        !! Requires flwr.server.server.Server to pass server.parameters to aggregate_fit(.)
        !! Requires src.server.client_managers.WeightedClientManager
    '''
    def __init__(self, client_valuation,
            *args, 
            metric=None,
            weighting_alpha=0.75, 
            start_softmax_temp=1.0,
            end_softmax_temp=1.0,
            rnd_softmax=0,
            softmax_temp_resets=[250,375], 
            start_client_weighting=0., 
            **kwargs):
        super().__init__(*args, **kwargs)
        self.client_valuation = client_valuation
        # if more of one metric is returned, a specific metric must be defined for weighting
        self.metric = metric

        # initialize global client weightings
        self.client_weightings = OrderedDict()
        for cid in range(self.config.simulation.num_clients):
            self.client_weightings[str(cid)] = start_client_weighting
        
        self.weighting_alpha = weighting_alpha

        # softmax temp hyperparameters
        self.softmax_linspace = deque(np.linspace(start_softmax_temp,end_softmax_temp,rnd_softmax))
        self.start_softmax_temp = start_softmax_temp
        self.end_softmax_temp = end_softmax_temp
        self.rnd_softmax = rnd_softmax
        self.softmax_temp_resets = softmax_temp_resets

    def configure_fit(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(rnd)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        # client weightings
        if rnd in self.softmax_temp_resets:
            self.softmax_linspace = deque(np.linspace(self.start_softmax_temp,self.end_softmax_temp,self.rnd_softmax))
        
        if len(self.softmax_linspace) == 0:
            softmax_temp = self.end_softmax_temp
        else:
            softmax_temp = self.softmax_linspace.popleft()
        client_prob_weights = list(softmax(np.array(list(self.client_weightings.values())), T=softmax_temp))

        if rnd % 10 == 0:
            for (k, v), p in zip(self.client_weightings.items(), client_prob_weights):
                logger.info(f'Client {k} (Weight|Likelihood): {round(v,4)} | {round(p,4)}')
            logger.info(f'Client weightings {pformat(self.client_weightings)}..')
            logger.info(f'Likelihood of sample client {pformat(client_prob_weights)}..')

        clients = client_manager.sample(
            num_clients=sample_size, client_weights=client_prob_weights, min_num_clients=min_num_clients
        )

        for c in clients:
            cid = c.cid
            self.ckp.log({f'client_{cid}_sample_%': client_prob_weights[int(cid)]}, step=rnd, commit=False)

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
        current_parameters: Parameters, 
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Update client weightings
        current_weights = parameters_to_weights(current_parameters)
        client_results = [
            (
                client.cid, 
                fit_res.num_examples,
                parameters_to_weights(fit_res.parameters)
            )
            for client, fit_res in results
        ]

        self.update_client_weightings(rnd, client_results, current_weights)

        # Convert results for fedavg
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]
        return weights_to_parameters(aggregate(weights_results)), {}

    def update_client_weightings(self, rnd:int, results: List[Tuple[int, int, Weights]], current_weights: Weights):
        round_config = self.on_fit_config_fn(rnd)
        assert 'lr' in round_config
        lr = round_config['lr']

        sampled_batch_weightings = [] # used for logging
        cu_w_val_metrics = self.client_valuation.evaluate(current_weights=current_weights, 
                                        lr=lr, 
                                        weights_1=[w for _, _, w in results], 
                                        weights_2=None,
                                        use_val=True, 
                                        client_samples=[s for _, s, _ in results])

        assert len(cu_w_val_metrics) == 1 or (self.metric is not None and self.metric in cu_w_val_metrics)

        if self.metric is None:
            # use the only metric
            k = next(iter(cu_w_val_metrics))
            cu_w_val_metrics = cu_w_val_metrics[k]
        else:
            #use user-specified metric
            cu_w_val_metrics = cu_w_val_metrics[self.metric]
        
        for client_value, (cid, _, _) in zip(cu_w_val_metrics, results):
            logger.debug(f'Client Value (CID {cid}): {client_value}')
            sampled_batch_weightings.append(client_value)
            self.client_weightings[cid] = (self.weighting_alpha) * self.client_weightings[cid] + (1 - self.weighting_alpha) * client_value

        # if rnd % 50 == 0:
        #     logger.info(f'Saving current client weightings {pformat(self.client_weightings)}..')
        #     self.ckp.save('results/client_weightings_{rnd}.pkl', self.client_weightings)
        
        # saving mean of client weightings
        sampled_client_weightings = [x for x in list(self.client_weightings.values()) if x]
        self.ckp.log({'mean_client_weighting': np.mean(sampled_client_weightings)}, step=rnd, commit=False)
        self.ckp.log({'mean_sample_batch_weighting': np.mean(sampled_batch_weightings)}, step=rnd, commit=False)
