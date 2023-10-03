from abc import ABC, abstractmethod
from src.log import Checkpoint
from flwr.server.server import Server
from src.utils import get_func_from_config

class App(ABC):
    '''
        Generic application class.

        Each application must contain of 
        1) a federated client function
        2) application-specific helper functions to be run on the server
        3) main federated learning pipeline
    '''    
    def __init__(self, ckp:Checkpoint, *args, **kwargs):
        self.ckp = ckp
        self.app_config = ckp.config.app

    def get_client_fn(self):
        client = get_func_from_config(self.app_config.client)
        def client_fn(cid: str):    
            return client(
                cid,
                self.ckp,
                **self.app_config.client.args
            )

        return client_fn

    def get_strategy_fns(self):
        # return the functions require to instantiate a Strategy
        return {
            'on_fit_config_fn': self.get_fit_config_fn(),
            'on_evaluate_config_fn': self.get_evaluate_config_fn(),
            'eval_fn': self.get_eval_fn()
        }

    @abstractmethod
    def get_fit_config_fn(self):
        '''
        returns a customizable client configuration for local training client.fit()
        Flower server strategy's on_fit_config_fn
        By default: (rnd: int) -> Dict[str, str]
        '''

    @abstractmethod
    def get_evaluate_config_fn(self):
        '''
        returns a customizable client configuration for personalized evaluation: client.evaluate()
        Flower server strategy's on_evaluate_config_fn
        By default: (rnd: int) -> Dict[str, str]
        '''

    @abstractmethod
    def get_eval_fn(self):
        '''
        returns a customizable centralized server evaluation
        Flower server strategy's eval_fn
        By default: (weights: fl.common.Weights) -> Optional[Tuple[float, float]]
        '''

    @abstractmethod
    def run(self, 
        server: Server):
        '''
            Application's federated learning main pipeline
        '''
        raise NotImplementedError


