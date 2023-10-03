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

class LeaveOneOut(ClientValuation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def evaluate(self, current_weights: Weights, weights_1: List[Weights], client_samples: List[int], **kwargs):
        all_client_updates = []
        for client_weights in weights_1:
            all_client_updates.append(
                get_updates(original_weights=current_weights, updated_weights=client_weights)
            )            
        total_examples = sum(client_samples)

        all_metrics = {}
        all_metrics['LOO'] = [] 

        inc_accuracy = None

        # for each client, remove client and do fedavg.
        for client_no in range(len(weights_1)):
            model = self.model

            tmp_num_examples = deepcopy(client_samples)
            tmp_client_updates = deepcopy(all_client_updates)

            # removing client from fedavg
            selected_client_update = tmp_client_updates.pop(client_no)
            selected_client_num_examples = tmp_num_examples.pop(client_no)
            selected_client_weighting = selected_client_num_examples / total_examples 
                
            # fedavg excluding client
            weights = deepcopy(current_weights)
            for client_update, client_num_example in zip(tmp_client_updates, tmp_num_examples):
                weighting = client_num_example / total_examples

                # update current weights
                weights = [weight + weighting * update.cpu().detach().numpy() for weight, update in zip(weights, client_update)]

            # set weights excluding client and run test
            set_weights(model, weights)
            _, ex_accuracy, _ = test(model, self.valloader, self.device, accuracy_scale=100.)
            
            if inc_accuracy is None:
                # only compute inc_accuracy once.
                weights = [weight + selected_client_weighting * update.cpu().detach().numpy() for weight, update in zip(weights, selected_client_update)]

                # run test including client (full fedavg for the round)
                set_weights(model, weights)
                _, inc_accuracy, _ = test(model, self.valloader, self.device, accuracy_scale=100.)

            all_metrics['LOO'].append(inc_accuracy - ex_accuracy)

        return all_metrics


