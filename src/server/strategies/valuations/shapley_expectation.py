import os
import torch
import numpy as np 
import random
import itertools
from copy import deepcopy

from src.apps.clients.client_utils import test
from src.models.model_utils import set_weights
from src.server.strategies.valuations import ClientValuation
from src.models.model_utils import get_updates

from typing import List
from flwr.common import Weights

import logging
logger = logging.getLogger(__name__)

class ShapleyExpectation(ClientValuation):
    # A Principled Approach to Data Valuation for Federated Learning
    def __init__(self, *args, epsilon=1.0, sigma=1.0, r=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        num_fit_clients = self.config.server.strategy.args.min_fit_clients
        self.T = round(2.0 * (r ** 2) / (epsilon ** 2) * np.log(2*num_fit_clients/sigma))
        logger.info(f'Number of permutations (T): {self.T}')

    def evaluate(self, current_weights: Weights, weights_1: List[Weights], client_samples: List[int], **kwargs):
        all_client_updates = []
        for client_weights in weights_1:
            all_client_updates.append(
                get_updates(original_weights=current_weights, updated_weights=client_weights)
            )  
        model = self.model

        # get previous round performance
        set_weights(model, current_weights) 
        _, u_prev, _ = test(model, self.valloader, self.device, accuracy_scale=100.)

        no_of_clients = len(weights_1)

        all_metrics = {}
        all_metrics['Shapley_Ex'] = [0.] * no_of_clients

        perms = list(itertools.permutations(range(no_of_clients)))
        sampled_perms = random.sample(perms, self.T) 
        for sampled_perm in sampled_perms:
            for client_no in range(no_of_clients):
                idx = sampled_perm.index(client_no)
                perm_idxs = sampled_perm[:idx]
            
                weights = deepcopy(current_weights)
                for perm_idx in perm_idxs:
                    client_update = all_client_updates[perm_idx]
                    client_num_example = client_samples[perm_idx]
                    weighting = client_num_example / sum(client_samples)

                    # update current weights
                    weights = [weight + weighting * update.cpu().detach().numpy() for weight, update in zip(weights, client_update)]

                set_weights(model, weights)
                _, acc, _ = test(model, self.valloader, self.device, accuracy_scale=100.)
                all_metrics['Shapley_Ex'][client_no] = all_metrics['Shapley_Ex'][client_no] + acc - u_prev
                u_prev = acc

        all_metrics['Shapley_Ex'] = [v / self.T for v in all_metrics['Shapley_Ex']]
        return all_metrics

