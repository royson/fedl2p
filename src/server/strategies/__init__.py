from .fedavg import FedAvg
from .resampling_fedavg import ResamplingFedAvg
from .sparse_fedavg import SparseFedAvg
from src.utils import get_func_from_config
from src.log import Checkpoint
from flwr.server.strategy import Strategy
from typing import Dict

def get_strategy(ckp: Checkpoint, strategy_fns: Dict[str, Strategy]):
    server_config = ckp.config.server
    strategy_fn = get_func_from_config(server_config.strategy)

    client_valuation = None
    if hasattr(server_config.strategy, 'valuation') and server_config.strategy.valuation is not None:
        valuation_fn = get_func_from_config(server_config.strategy.valuation)
        client_valuation = valuation_fn(ckp=ckp, **server_config.strategy.valuation.args)

    return strategy_fn(
        client_valuation=client_valuation,
        ckp=ckp,
        fraction_fit=float(server_config.strategy.args.min_fit_clients) / ckp.config.simulation.num_clients,
        fraction_eval=1.0,
        min_available_clients=ckp.config.simulation.num_clients, # all clients are available  
        min_eval_clients=ckp.config.simulation.num_clients,
        **server_config.strategy.args,
        **strategy_fns
    )