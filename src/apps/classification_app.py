import flwr as fl
from flwr.server.server import Server
from flwr.server.history import History
from flwr.common import Parameters
from matplotlib import backend_tools
import torch
import os
from copy import deepcopy
from src.apps import App
from src.apps.clients import test
from src.utils import get_func_from_config
from src.apps.app_utils import update_lr
from src.models.model_utils import set_weights
from typing import Dict, Callable, Optional, Tuple 
import numpy as np
# import timeit
import random
import math

import logging
logger = logging.getLogger(__name__)

class ClassificationApp(App):    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        last_round = self.ckp.load('models/last_round_saved.pkl')
        self.load = False
        self.start_run = 1

        if 'load' in self.ckp.config and os.path.exists(self.ckp.config.load):
            logger.info(f'Start from pretrained model: {self.ckp.config.load}')  
            self.current_weights = self.ckp.offline_load(self.ckp.config.load) 
            self.load = True         
        elif last_round is not None:
            self.current_weights = self.ckp.load(f'models/latest_weights.pkl')
            self.start_run = last_round + 1
            logger.info(f'Starting from round {self.start_run}..')
        

    def get_fit_config_fn(self):
        """Return a configuration with static batch size and (local) epochs."""
        def fit_config_fn(rnd: int) -> Dict[str, str]:
            fit_config = self.ckp.config.app.on_fit
            steps = fit_config.lr_decay.steps
            factors = fit_config.lr_decay.factors

            # if steps aren't empty and an equal number of factors are given
            # then use step LR decay, else default to exponential LR decay
            if len(steps) > 0 and len(factors) == len(steps):
                gamma = 1.0 # this will make the EXP LR DECAY inside each client to do nothing
                current_lr = fit_config.start_lr
                for step, factor in zip(steps, factors):
                    if rnd > step:
                        current_lr *= factor
            else:
                current_lr, gamma = update_lr(
                    current_round=rnd,
                    total_rounds=self.ckp.config.app.run.num_rounds,
                    start_lr=fit_config.start_lr,
                    end_lr=fit_config.end_lr,
                )

            self.ckp.log({"global_LR": current_lr}, step=rnd)

            client_config = {
                "lr": current_lr,
                "current_round": rnd,
                "gamma": gamma}
            return client_config

        return fit_config_fn

    def get_evaluate_config_fn(self):
        """"Client evaluate. Evaluate on client's test set"""
        def evaluate_config_fn(rnd: int) -> Dict[str, str]:
            eval_config = self.ckp.config.app.on_evaluate

            client_config = {
                "lr": eval_config.lr,
                "current_round": rnd,
                "finetune_epochs": eval_config.finetune_epochs,
                "freeze_bn_buffer": eval_config.freeze_bn_buffer }
            return client_config

        return evaluate_config_fn

    def get_eval_fn(self) -> Callable[[fl.common.Weights], Optional[Tuple[float, float]]]:
        """Return an evaluation function for centralized evaluation."""
        def evaluate(weights: fl.common.Weights, partition: str) -> Optional[Tuple[float, float]]:
            # determine device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

            # instantiate model
            net_config = self.ckp.config.models.net
            data_config = self.ckp.config.data
            arch_fn = get_func_from_config(net_config)
            model = arch_fn(device=device, **net_config.args)

            # instantiate dataloader
            data_class = get_func_from_config(data_config)
            dataset = data_class(self.ckp, **data_config.args)

            # assume net is always in the front of the list
            set_weights(model, weights)
            model.to(device)
            
            loss, accuracy, loss_clt, accuracy_clt = 0,0,0,0

            if self.ckp.config.app.eval_fn.centralized_eval:
                testloader = dataset.get_dataloader(
                                               data_pool='server',
                                               partition=partition,
                                               batch_size=self.ckp.config.app.eval_fn.batch_size,
                                               augment=False,
                                               num_workers=0)
                loss, accuracy, _num_samples = test(model, testloader, device=device)

                metrics = {f"centralized_{partition}_loss": loss, 
                    f"centralized_{partition}_acc": accuracy * 100}

            else:
                # if no data on server, evaluate on test data on train pool.                
                list_clients = list(range(self.ckp.config.simulation.num_clients))
                
                losses_fl, accuracies_fl, num_samples_fl = [], [], []
                for client_id in list_clients:
                    
                    testloader = dataset.get_dataloader(
                       data_pool='server',
                       partition = partition,
                       batch_size = self.ckp.config.app.eval_fn.batch_size,
                       num_workers = 0,
                       augment=False,
                       cid=str(client_id),
                    )
                
                    _loss, _acc, _num_samples = test(model, testloader, device=device)
                    losses_fl.append(_loss)
                    accuracies_fl.append(_acc)
                    num_samples_fl.append(_num_samples)

                loss_clt = np.average(losses_fl, weights=num_samples_fl)
                accuracy_clt = np.average(accuracies_fl, weights=num_samples_fl)
            

                metrics = {
                    f"client_side_average_{partition}_loss": loss_clt,
                    f"client_side_average_{partition}_acc": accuracy_clt * 100}

            del model
            
            return loss, metrics

        return evaluate

    def run(self, server: Server):
        """Run federated averaging for a number of rounds."""
        history = History()

        def _centralized_evaluate(rnd, partition, log=True):
            server_metrics = None
            # Evaluate model using strategy implementation (runs eval_fn)
            parameters = server.parameters 
            res_cen = server.strategy.evaluate(parameters=parameters, partition=partition)
            if res_cen is not None:
                loss_cen, server_metrics = res_cen
                history.add_loss_centralized(rnd=rnd, loss=loss_cen)
                history.add_metrics_centralized(rnd=rnd, metrics=server_metrics)
                if log:
                    self.ckp.log(server_metrics, step=rnd)
            return server_metrics

        # Initialize parameters
        if self.load or self.start_run > 1:
            server.parameters = self.current_weights
            logger.info('[*] Global Parameters Loaded.')
        else:
            server.parameters = server._get_initial_parameters()
            # Get initial test accuracy
            server_metrics = _centralized_evaluate(0, 'test')

        # Run federated learning for num_rounds
        logger.info("FL starting")
        # start_time = timeit.default_timer()

        app_run_config = self.ckp.config.app.run

        for rnd in range(self.start_run, app_run_config.num_rounds + 1):
            # Train model and replace previous global model
            server_metrics = None
            clients_metrics = None
            res_fit = server.fit_round(rnd=rnd)
            if res_fit:
                parameters_prime, _, (results, _) = res_fit  # fit_metrics_aggregated
                clients_metrics = [res[1].metrics for res in results]

                if parameters_prime:
                    server.parameters = parameters_prime

            if rnd % app_run_config.test_every_n == 0:
                logger.debug(f"[Round {rnd}] Evaluating global model on test set.")
                server_metrics = _centralized_evaluate(rnd, 'test')
                logger.info(f"[Round {rnd}] {server_metrics}")

            # end of round saving
            self.ckp.save(f'results/round_{rnd}.pkl', 
                {'round': rnd,
                'clients_metrics': clients_metrics, 
                'server_metrics': server_metrics})
            self.ckp.save(f'models/latest_weights.pkl',
                server.parameters)
            if rnd == self.start_run or rnd % 10 == 0:
                # save model every 10 rounds
                self.ckp.save(f'models/weights_round_{rnd}.pkl',
                    server.parameters)
            self.ckp.save(f'models/last_round_saved.pkl', rnd)

        # test set evaluation and logging using wandb summary metrics
        logger.info(f"[Round {rnd}] Training done. Final test evaluation")
        server_metrics = _centralized_evaluate(rnd, 'test', log=False)
        logger.info(f"Final Test Result: {server_metrics}")
        for k, v in server_metrics.items():
            self.ckp.log_summary(k, v)

        logger.info(f'Running personalized FL pipeline')
        personalized_metrics = server.evaluate_round(rnd)
        loss_aggregated, metrics_aggregated, _ = personalized_metrics
        
        # save the final tests result
        for k,v in metrics_aggregated.items():
            logger.info(f'Logging {k}:{v}')
            self.ckp.log_summary(k, v)
        self.ckp.log_summary(f'ps_FL_loss', loss_aggregated)

        self.ckp.save(f'results/round_{rnd}_test.pkl', 
                {'server_metrics': server_metrics})
        self.ckp.save(f'models/weights_{rnd}_final.pkl',
                server.parameters)
