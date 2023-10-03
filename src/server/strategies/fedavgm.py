from curses import start_color
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.common.logger import log
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate
from src.server.strategies import FedAvg
import logging
import numpy as np
logger = logging.getLogger(__name__)


WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_eval_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_eval_clients`.
"""


class FedAvgM(FedAvg):
    """Configurable FedAvg with Momentum strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes
    def __init__(
        self,
        total_rnd: int,
        *args,
        #fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        #evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        server_learning_rate: float = 1.0,
        server_momentum: float = 0.0,
        server_start_lr: float = 1.0,
        server_end_lr: float = None,
        **kwargs,
    ) -> None:
        """Federated Averaging with Momentum strategy.
        Implementation based on https://arxiv.org/pdf/1909.06335.pdf
        Parameters
        ----------
        server_learning_rate: float
            Server-side learning rate used in server-side optimization.
            Defaults to 1.0.
        server_momentum: float
            Server-side momentum factor used for FedAvgM. Defaults to 0.0.
        """

        super().__init__(*args,**kwargs)
        self.server_learning_rate = server_learning_rate
        self.server_momentum = server_momentum
        self.server_opt: bool = (self.server_momentum != 0.0) or (
            self.server_learning_rate != 1.0
        )
        self.total_rnd = total_rnd
        self.server_start_lr = server_start_lr
        self.server_end_lr = server_end_lr
        self.momentum_vector: Optional[Weights] = None
        
        #self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        #self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn

    def __repr__(self) -> str:
        rep = f"FedAvgM(accept_failures={self.accept_failures})"
        return rep

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        return self.initial_parameters
    
    def update_server_lr(
        self,
        rnd: int,
    ) -> float:
        ''' update server lr the same as update_lr'''
        if self.server_end_lr:
            gamma = np.power(self.server_end_lr / self.server_start_lr, 1.0 / self.total_rnd)
            current_lr = self.server_start_lr * np.power(gamma, rnd)
        else:
            current_lr = self.server_learning_rate
        logger.info(f'Current server learning rate = {current_lr}')
        return current_lr
 
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
        # Convert results
        weights_results = [
            (parameters_to_weights(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        fedavg_result = aggregate(weights_results)
        # following convention described in
        # https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
        


        if self.server_opt:
            # You need to initialize the model
            assert (
                current_parameters is not None
            ), "When using server-side optimization, model needs to be initialized."
            #initial_weights = parameters_to_weights(current_parameters)

            # get current server lr
            current_lr = self.update_server_lr(rnd)
            # remember that updates are the opposite of gradients
            pseudo_gradient = [
                x - y
                for x, y in zip(
                    parameters_to_weights(current_parameters), fedavg_result
                )
            ]
            if self.server_momentum > 0.0:
                if rnd > 1:
                    assert (
                        self.momentum_vector
                    ), "Momentum should have been created on round 1."
                    self.momentum_vector = [
                        self.server_momentum * x + (1-self.server_momentum) * y
                        for x, y in zip(self.momentum_vector, pseudo_gradient)
                    ]
                else:
                    self.momentum_vector = pseudo_gradient

                # No nesterov for now
                pseudo_gradient = self.momentum_vector

            # SGD
            fedavg_result = [
                x - current_lr * y
                for x, y in zip(parameters_to_weights(current_parameters), pseudo_gradient)
            ]
            # Update current weights
            #self.initial_parameters = weights_to_parameters(fedavg_result)

        parameters_aggregated = weights_to_parameters(fedavg_result)

        '''
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif rnd == 1:
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        '''
        #return parameters_aggregated, metrics_aggregated
        return parameters_aggregated, {}