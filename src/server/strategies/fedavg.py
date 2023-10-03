import numpy as np
from flwr.server.strategy import FedAvg as FlowerFedAvg
from flwr.server.client_manager import ClientManager
from src.utils import get_func_from_config
from pprint import pformat
from collections import defaultdict

from typing import Dict, Optional, Tuple, List
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
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

class FedAvg(FlowerFedAvg):
    '''
        Same as flwr.server.strategy.FedAvg but we modify the evaluate() function as 
        that would allow to distinguish between an evaluation of validation/tests set

        + extra logging :)

        !! Requires src.server.Server to pass server.parameters to aggregate_fit(.)
    '''
    def __init__(self, ckp, client_valuation, *args, log=False, aggregate_equal_weights=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.ckp = ckp
        self.config = ckp.config
        self.client_valuation = client_valuation
        self.log = log
        self.aggregate_equal_weights = aggregate_equal_weights

        data_config = ckp.config.data
        self.validation_set = hasattr(data_config.args, 'val_ratio') and data_config.args.val_ratio > 0.
        data_class = get_func_from_config(data_config)
        dataset = data_class(ckp, **data_config.args)
        self.test_alpha = dataset.test_alpha

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

        # Convert results for fedavg
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples if not self.aggregate_equal_weights else 1)
            for client, fit_res in results
        ]

        aggregated_weights = aggregate(weights_results)

        # DEBUGGING
        # for w in aggregated_weights:
        #     if np.isnan(w).any() or np.isinf(w).any():
        #         import pdb
        #         pdb.set_trace()


        # log training loss
        train_summary = defaultdict(list)
        for _, fit_res in results:
            for m, v in fit_res.metrics.items():
                train_summary[m].append(v)
        for k, v in train_summary.items():
            self.ckp.log({f'mean_{k}': np.mean(v)}, step=rnd, commit=False)
        
        if self.log:
            current_weights = parameters_to_weights(current_parameters)

            client_cids = [client.cid for client, _ in results]
            client_weights = [parameters_to_weights(fit_res.parameters) for _, fit_res in results]
            client_samples = [fit_res.num_examples for _, fit_res in results]

            self.log_round(rnd, client_cids, client_samples, client_weights, aggregated_weights, current_weights)

        return weights_to_parameters(aggregated_weights), {}

    def log_round(self, rnd:int, client_cids: List[int], client_samples: List[int], client_weights: List[Weights], aggregated_weights: Weights, current_weights: Weights):
        round_config = self.on_fit_config_fn(rnd)
        assert 'lr' in round_config
        lr = round_config['lr']

        all_metrics = []

        if self.validation_set:
            cu_w_val_metrics = self.client_valuation.evaluate(current_weights=current_weights, 
                                            lr=lr, 
                                            weights_1=client_weights, 
                                            weights_2=None,
                                            use_val=True, 
                                            client_samples=client_samples)
            all_metrics.append(('CU_w_Val', cu_w_val_metrics))

        
        cu_w_au_metrics = self.client_valuation.evaluate(current_weights=current_weights, 
                                        lr=lr, 
                                        weights_1=client_weights, 
                                        weights_2=aggregated_weights,
                                        use_val=False, 
                                        client_samples=client_samples)
        all_metrics.append(('CU_w_AU', cu_w_au_metrics))

        # au_w_val_metrics = self.client_valuation.evaluate(current_weights=current_weights, 
        #                                 lr=lr, 
        #                                 weights_1=[aggregated_weights], 
        #                                 weights_2=None,
        #                                 use_val=True, 
        #                                 client_samples=client_samples)
        # all_metrics.append(('AU_w_Val', au_w_val_metrics))

        for name, metrics in all_metrics:
            for k, v in metrics.items(): 
                if type(v[0]) != float:
                    # log mean across clients for each class. v is [client1, client2, client3,...] where client1=[value for each class]
                    tmp = {}
                    for class_id, class_v in enumerate(list(zip(*v))):
                        if len(class_v) > 1:
                            log_name = f'{name} | mean_{k}'
                            tmp[f'{log_name} | Class{class_id}'] = np.mean(class_v)
                        else:
                            log_name = f'{name} | {k}'
                            tmp[f'{log_name} | Class{class_id}'] = class_v[0]

                    tmp[log_name] = np.mean([v for v in tmp.values()])

                    self.ckp.log(tmp, step=rnd, commit=False)
                else:
                    # log mean across clients for each metric
                    if len(v) > 1:
                        self.ckp.log({f'{name} | mean_{k}': np.mean(v)}, step=rnd, commit=False)
                    else:
                        self.ckp.log({f'{name} | {k}': v[0]}, step=rnd, commit=False)

    def evaluate(
        self, parameters: Parameters, partition: str = 'test',
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate model parameters using an evaluation function."""
        if self.eval_fn is None:
            # No evaluation function provided
            return None
        weights = parameters_to_weights(parameters)
        eval_res = self.eval_fn(weights, partition)
        if eval_res is None:
            return None
        loss, other = eval_res
        if isinstance(other, float):
            metrics = {"accuracy": other}
        else:
            metrics = other
        return loss, metrics


    def configure_evaluate(
        self, rnd: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(rnd)
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        if rnd >= 0:
            sample_size, min_num_clients = self.num_evaluation_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients, all_available=True
            )
        else:
            clients = list(client_manager.all().values())

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    def aggregate_evaluate(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        client_results = {}
        for client, evaluate_res in results:
            client_results[client.cid] = (
                evaluate_res.num_examples,
                evaluate_res.loss,
                evaluate_res.metrics,
            )

        loss_aggregated, accuracy_results = weighted_loss_avg(
            client_results,
            self.test_alpha
        )
        return loss_aggregated, accuracy_results

def weighted_loss_avg(results: Dict[str, Tuple[int, float, Optional[Dict[str, float]]]], personalized_fl_groups: Dict[str, int]) -> Tuple[float, float]:
    """Aggregate evaluation results obtained from multiple clients.
    TODO: rename variables to include other metrics apart from accuracy. 
    """
    accuracy_results = {}
    if personalized_fl_groups is not None and len(personalized_fl_groups) > 1:
        from_id = 0
        for group, to_id in personalized_fl_groups.items():
            group_examples = 0
            group_correct_preds = defaultdict(float)
            group_loss = 0
            for cid in range(from_id, from_id + int(to_id)):
                num_examples, loss, metrics = results[str(cid)]
                group_examples += num_examples
                for k, acc in metrics.items():
                    if 'test_acc' in k or 'accuracy' in k:
                        group_correct_preds[k] += num_examples * acc
                    else:
                        group_correct_preds[k] += acc
                group_loss += num_examples * loss
            from_id += to_id
            for k, v in group_correct_preds.items():
                if 'test_acc' in k or 'accuracy' in k:
                    accuracy_results[f'ps_{k}_alpha{group}({to_id} clients)'] = v / group_examples * 100
                else:
                    accuracy_results[f'mean_{k}_alpha{group}({to_id} clients)'] = v / float(to_id)
    
    # overall accuracy    
    num_total_evaluation_examples = sum(
        [num_examples for num_examples, _, _ in results.values()]
    )
    weighted_losses = [num_examples * loss for num_examples, loss, _ in results.values()]
    num_correct_preds = defaultdict(list)
    for num_examples, _, metrics in results.values():
        for k, acc in metrics.items():
            if 'test_acc' in k or 'accuracy' in k:
                num_correct_preds[k].append(num_examples * acc)
            else:
                num_correct_preds[k].append(acc)

    # num_correct_preds = [num_examples * accuracy for num_examples, _, accuracy in results.values()]
    for k, v in num_correct_preds.items():
        if 'test_acc' in k or 'accuracy' in k:
            accuracy_results[f'ps_{k}'] =  sum(v) / num_total_evaluation_examples * 100
        else:
            accuracy_results[f'mean_{k}'] = np.mean(v)


    return sum(weighted_losses) / num_total_evaluation_examples, accuracy_results