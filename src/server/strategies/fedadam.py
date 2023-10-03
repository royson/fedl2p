from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    Weights,
    parameters_to_weights,
    weights_to_parameters,
)
from flwr.server.client_proxy import ClientProxy

from .fedopt import FedOpt


class FedAdam(FedOpt):
    """Adaptive Federated Optimization using Adam (FedAdam) [Reddi et al.,
    2020] strategy.
    Paper: https://arxiv.org/abs/2003.00295
    """
    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
        ckp,
        *,
        fraction_fit: float = 0.1,
        fraction_eval: float = 0.1,
        min_fit_clients: int = 2,
        min_eval_clients: int = 2,
        min_available_clients: int = 2,
        eval_fn: Optional[
            Callable[[Weights], Optional[Tuple[float, Dict[str, Scalar]]]]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Parameters = None,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        tau: float = 1e-3,
    ) -> None:
        """Federated learning strategy using Adagrad on server-side.
        Implementation based on https://arxiv.org/abs/2003.00295
        Args:
            fraction_fit (float, optional): Fraction of clients used during
                training. Defaults to 0.1.
            fraction_eval (float, optional): Fraction of clients used during
                validation. Defaults to 0.1.
            min_fit_clients (int, optional): Minimum number of clients used
                during training. Defaults to 2.
            min_eval_clients (int, optional): Minimum number of clients used
                during validation. Defaults to 2.
            min_available_clients (int, optional): Minimum number of total
                clients in the system. Defaults to 2.
            eval_fn (Callable[[Weights], Optional[Tuple[float, float]]], optional):
                Function used for validation. Defaults to None.
            on_fit_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure training. Defaults to None.
            on_evaluate_config_fn (Callable[[int], Dict[str, str]], optional):
                Function used to configure validation. Defaults to None.
            accept_failures (bool, optional): Whether or not accept rounds
                containing failures. Defaults to True.
            initial_parameters (Parameters): Initial set of parameters from the server.
            eta (float, optional): Server-side learning rate. Defaults to 1e-1.
            eta_l (float, optional): Client-side learning rate. Defaults to 1e-1.
            beta_1 (float, optional): Momentum parameter. Defaults to 0.9.
            beta_2 (float, optional): Second moment parameter. Defaults to 0.99.
            tau (float, optional): Controls the algorithm's degree of adaptability.
                Defaults to 1e-9.
        """
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_eval=fraction_eval,
            min_fit_clients=min_fit_clients,
            min_eval_clients=min_eval_clients,
            min_available_clients=min_available_clients,
            eval_fn=eval_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            eta=eta,
            eta_l=eta_l,
            beta_1=beta_1,
            beta_2=beta_2,
            tau=tau,
        )
        self.delta_t: Optional[Weights] = None
        self.v_t: Optional[Weights] = None
        self.ckp = ckp
        self.config = ckp.config

    def __repr__(self) -> str:
        rep = f"FedAdam(accept_failures={self.accept_failures})"
        return rep

    def aggregate_fit(
        self,
        rnd: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        fedavg_parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            rnd=rnd, results=results, failures=failures
        )
        if fedavg_parameters_aggregated is None:
            return None, {}
        fedavg_weights_aggregate = parameters_to_weights(fedavg_parameters_aggregated)
        aggregated_updates = [
            x - y for x, y in zip(fedavg_weights_aggregate, self.current_weights)
        ]
        # Adam
        if not self.delta_t:
            self.delta_t = [np.zeros_like(x) for x in self.current_weights]
        self.delta_t = [
            self.beta_1 * x + (1.0 - self.beta_1) * y
            for x, y in zip(self.delta_t, aggregated_updates)
        ]
        if not self.v_t:
            self.v_t = [np.zeros_like(x) for x in self.delta_t]
        self.v_t = [
            self.beta_2 * x + (1.0 - self.beta_2) * np.multiply(y, y)
            for x, y in zip(self.v_t, self.delta_t)
        ]
        new_weights = [
            x + self.eta * y / (np.sqrt(z) + self.tau)
            for x, y, z in zip(self.current_weights, self.delta_t, self.v_t)
        ]
        self.current_weights = new_weights
        #return weights_to_parameters(self.current_weights), metrics_aggregated
        return weights_to_parameters(self.current_weights), {}
    
    def evaluate(
        self, parameters: Parameters, partition: str = 'val',
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