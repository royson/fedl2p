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

class ShapleyMC(ClientValuation):
    # Game of Gradients: Mitigating Irrelevant Clients in Federated Learning
    def __init__(self, num_permutations, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_permutations = num_permutations

    def evaluate(self, current_weights: Weights, weights_1: List[Weights], client_samples: List[int], **kwargs):
        all_client_updates = []
        for client_weights in weights_1:
            all_client_updates.append(
                get_updates(original_weights=current_weights, updated_weights=client_weights)
            )  

        all_metrics = {}
        all_metrics['Shapley_MC'] = []

        no_of_clients = len(weights_1)

        for client_no in range(no_of_clients):
            perms = list(itertools.permutations(range(no_of_clients)))
            sampled_perms = random.sample(perms, self.num_permutations)

            model = self.model

            shapley_value = 0.
            for sampled_perm in sampled_perms:
                idx = sampled_perm.index(client_no)
                perm_idxs = sampled_perm[:idx]
                
                weights = deepcopy(current_weights)
                for perm_idx in perm_idxs:
                    client_update = all_client_updates[perm_idx]
                    client_num_example = client_samples[perm_idx]
                    weighting = client_num_example / sum(client_samples)

                    # update current weights
                    weights = [weight + weighting * update.cpu().detach().numpy() for weight, update in zip(weights, client_update)]

                # set weights excluding client and run test
                set_weights(model, weights)
                _, ex_accuracy, _ = test(model, self.valloader, self.device, accuracy_scale=100.)
                
                # include client
                client_update = all_client_updates[client_no]
                client_num_example = client_samples[client_no]
                weighting = client_num_example / sum(client_samples)
                weights = [weight + weighting * update.cpu().detach().numpy() for weight, update in zip(weights, client_update)]

                set_weights(model, weights)
                _, inc_accuracy, _ = test(model, self.valloader, self.device, accuracy_scale=100.)

                # run test including client
                v = inc_accuracy - ex_accuracy

                shapley_value += v

            all_metrics['Shapley_MC'].append(shapley_value / len(sampled_perms))
        
        return all_metrics
