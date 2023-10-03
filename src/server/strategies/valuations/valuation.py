import os
import torch
from abc import ABC, abstractmethod
from src.utils import get_func_from_config
from src.log import Checkpoint

class ClientValuation(ABC):
    def __init__(self, model_name, ckp:Checkpoint, *args, partition='val', val_batch_size=32, **kwargs):
        self.ckp = ckp
        self.config = ckp.config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert model_name in self.config.models.keys()
        net_config = self.config.models[model_name]
        arch_fn = get_func_from_config(net_config)
        self.model = arch_fn(**net_config.args)
        self.net_length = len(self.model.state_dict().keys())

        self.no_of_classes = self.model.num_classes

        data_config = self.config.data
        self.validation_set = hasattr(data_config.args, 'val_ratio') and data_config.args.val_ratio > 0.
        self.dataloader = None
        self.valloader = None
        if self.validation_set:            
            data_class = get_func_from_config(data_config)

            self.dataloader = data_class(self.ckp, **data_config.args).get_dataloader

            self.valloader = self.dataloader(
                data_pool='train',
                partition=partition, 
                batch_size=val_batch_size, 
                num_workers=0
            )

    @abstractmethod
    def evaluate(self, *args, **kwargs):
        '''
            Returns a dict of list of distance values or list of distance values per class
        '''