import sys
from pprint import pformat
from config import get_args
from pathlib import Path
from src.data import prepare_fl_partitioned_dataset
from src.apps import get_app
from src.server import get_server
from src.simulation import start_simulation
from src.log import Checkpoint

import numpy as np
import random
import torch

import logging
logger = logging.getLogger(__name__)

def main():
    # YAML & Arguments Config Passing 
    config = get_args()
    # Initialize logger
    ckp = Checkpoint(config)

    logger.info(f'Command ran: {" ".join(sys.argv)}')
    logger.info(pformat(config))
    if config['run']:
        # Initialize wanDB. You can access ckp.config as attributes.
        ckp.init_wandb()
        logger.info(f'Run ID: {ckp.config.run_id}')
        
        # Setting seed for reproducibility
        if hasattr(ckp.config, 'seed'):
            logger.info(f'Setting fixed seed: {ckp.config.seed}')
            torch.manual_seed(ckp.config.seed)
            random.seed(ckp.config.seed)
            np.random.seed(ckp.config.seed)

        torch.set_num_threads(ckp.config.cpu_threads)

        # Downloads and partitions dataset federated-ly. 
        # Federated directory is accessible via ckp.config.data.fed_dir
        prepare_fl_partitioned_dataset(ckp)

        # Application-specific code
        # Contains 1) client pipeline, 2) how it's evaluated on the server, 3) FL main pipeline
        my_app = get_app(ckp)

        # Get Server, which consists of Strategy & ClientManager, using App's strategy_fns namely,
        # on_fit_config_fn, on_evaluate_config_fn, eval_fn
        server = get_server(ckp, strategy_fns=my_app.get_strategy_fns())
        
        # Initialize ray & flwr annd call app.run()
        start_simulation(ckp, server=server, app=my_app)

        # Sync local checkpoint config with wandb config
        ckp.update_wandb_config() 

if __name__ == "__main__":
    main()
