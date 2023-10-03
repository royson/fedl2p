import os
import torch
import numpy as np 

from collections import defaultdict
from src.utils import get_func_from_config
from src.models.model_utils import set_weights, get_updates
from src.server.strategies import FedAvg

from typing import Dict, Optional, Tuple, List
from flwr.server.client_proxy import ClientProxy
from flwr.common import (
    Parameters,
    Weights,
    Scalar,
    FitRes,
    parameters_to_weights,
    weights_to_parameters,
)

import logging
logger = logging.getLogger(__name__)

class SparseFedAvg(FedAvg):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.ckp and self.config is defined in parent class
        net_config = self.config.models.net
        arch_fn = get_func_from_config(net_config)
        self.net = arch_fn(**net_config.args)

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

        current_weights = parameters_to_weights(current_parameters)
        # Convert results for fedavg
        weights_results = [
            (   dict(
                    zip(
                        list(self.net.state_dict().keys()),
                        get_updates(original_weights=current_weights, 
                                updated_weights=parameters_to_weights(fit_res.parameters))
                    )
                ),
                fit_res.num_examples
            )
            for client, fit_res in results
        ]

        aggregated_weights = self.aggregate(current_weights=current_weights, results=weights_results, rnd=rnd)

        if self.log:
            # logging
            client_cids = [client.cid for client, _ in results]
            client_weights = [parameters_to_weights(fit_res.parameters) for _, fit_res in results]
            client_samples = [fit_res.num_examples for _, fit_res in results]

            self.log_round(rnd, client_cids, client_samples, client_weights, aggregated_weights, current_weights)

        return weights_to_parameters(aggregated_weights), {}

    def aggregate(self, current_weights: Weights, results: List[Tuple[Dict[str, Weights], int]], rnd: int) -> Weights:
        num_examples_total = sum([num_examples for _, num_examples in results])
        num_clients = len(results)

        client_layer_mask = defaultdict(int)
        client_layer_count = defaultdict(int)
        layer_updates = defaultdict(list)
        # client_layer_mag = defaultdict(int) #tmp
        # client_layer_max = defaultdict(int) #tmp
        for client_updates, no_of_samples in results:
            for layer_name, layer_update in client_updates.items():
                layer_updates[layer_name].append((no_of_samples / num_examples_total) * layer_update)
                if 'fc' in layer_name or 'conv' in layer_name:
                    threshold = torch.mean(layer_update).item()
                    layer_thres = (torch.abs(layer_update) > threshold).long() # if parameter fail threshold, add count
                    layer_sign = torch.sign(layer_update) * layer_thres # get the gradient direction of parameters that fail threshold
                    client_layer_mask[layer_name] += layer_sign
                    client_layer_count[layer_name] += layer_thres
                    # client_layer_mag[layer_name] += layer_update.norm(2).item()
                    # client_layer_max[layer_name] += torch.max(layer_update).item()
        
        for layer_name in client_layer_mask.keys():
            client_layer_mask[layer_name] = torch.abs(client_layer_mask[layer_name]) == client_layer_count[layer_name]
            # logger.info(f'{layer_name} Mean L2: {client_layer_mag[layer_name] / len(results)}, Mean Max: {client_layer_max[layer_name] / len(results)}')
            self.ckp.log({f'{layer_name} Non-Conflicting Grads (%)': 
                            torch.sum(client_layer_mask[layer_name]) / np.prod(client_layer_mask[layer_name].size()).item() * 100.}
                            ,step=rnd, commit=False)
                
        updates_prime = []
        for layer_name in layer_updates.keys():
            u = torch.sum(torch.stack(layer_updates[layer_name]), dim=0)
            if layer_name in client_layer_mask:
                updates_prime.append(client_layer_mask[layer_name] * u)
            else:
                updates_prime.append(u)

        assert len(current_weights) == len(updates_prime)

        return [current_weight + update.cpu().detach().numpy() for current_weight, update in zip(current_weights, updates_prime)]
        