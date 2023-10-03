from .server import Server
from src.utils import get_func_from_config
from src.server.strategies import get_strategy

def get_server(ckp, strategy_fns):
    # Server consists of flwr.server.strategy.Strategy and flwr.server.client_manager.ClientManager
    server_config = ckp.config.server
    server_fn = get_func_from_config(server_config)

    strategy = get_strategy(ckp, strategy_fns)

    client_manager = get_func_from_config(server_config.client_manager)(ckp)

    return server_fn(strategy=strategy, client_manager=client_manager)

