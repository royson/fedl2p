# ! This code has been copied from https://github.com/adap/flower/blob/main/src/py/flwr/server/strategy/fedopt.py commit  59a32d5
# and removed the annoying requirement of having to pass the initialized model at construction time

from tkinter import E
from typing import Callable, Dict, Optional, Tuple

from flwr.common import Parameters, Scalar, Weights, parameters_to_weights

from flwr.server.strategy import FedAvg

import pickle

class FedOpt(FedAvg):
    """Configurable FedAdagrad strategy implementation."""

    # pylint: disable=too-many-arguments,too-many-instance-attributes,too-many-locals
    def __init__(
        self,
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
        initial_parameters: Parameters,
        eta: float = 1e-1,
        eta_l: float = 1e-1,
        beta_1: float = 0.0,
        beta_2: float = 0.0,
        tau: float = 1e-9,
    ) -> None:
        """Federated Optim strategy interface.
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
            beta_1 (float, optional): Momentum parameter. Defaults to 0.0.
            beta_2 (float, optional): Second moment parameter. Defaults to 0.0.
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
        )

        if initial_parameters:
            self.current_weights = parameters_to_weights(initial_parameters)
        else:
            self.current_weights = None
            # ! this will trigger a crash if the user doesn't copy the sever weights before the 1st round begin
        print('type of current weights:',type(self.current_weights))
        self.eta = eta
        self.eta_l = eta_l
        self.tau = tau
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def __repr__(self) -> str:
        rep = f"FedOpt(accept_failures={self.accept_failures})"
        return rep