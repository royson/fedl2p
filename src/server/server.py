from flwr.server.server import Server as FlowerServer
from flwr.server.server import FitResultsAndFailures, fit_clients
from typing import Tuple, Optional, Dict, Union
from flwr.common.logger import log
from logging import DEBUG, INFO, WARNING
from flwr.common import Parameters, Scalar

class Server(FlowerServer):
    """
    Similar to flwr.server.server.Server but pass the current round parameters
    as an argument to aggregate clients' parameters.

    Uses flower's logger.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit_round(
        self, rnd: int
    ) -> Optional[
        Tuple[Optional[Parameters], Dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_fit(
            rnd=rnd, parameters=self.parameters, client_manager=self._client_manager
        )

        if not client_instructions:
            log(INFO, "fit_round: no clients selected, cancel")
            return None
        log(
            DEBUG,
            "fit_round: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions
        )
        log(
            DEBUG,
            "fit_round received %s results and %s failures",
            len(results),
            len(failures),
        )

        if len(failures) > 0:
            log(INFO, f'{len(failures)} clients failed.')
            import sys
            sys.exit(0)

        # Aggregate training results
        aggregated_result: Union[
            Tuple[Optional[Parameters], Dict[str, Scalar]],
            Optional[Weights],  # Deprecated
        ] = self.strategy.aggregate_fit(rnd, results, failures, current_parameters=self.parameters)

        metrics_aggregated: Dict[str, Scalar] = {}
        parameters_aggregated, metrics_aggregated = aggregated_result

        return parameters_aggregated, metrics_aggregated, (results, failures)
