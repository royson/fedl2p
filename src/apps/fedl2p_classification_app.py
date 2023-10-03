import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import math
from flwr.server.server import Server
from flwr.server.history import History
from flwr.common import parameters_to_weights, weights_to_parameters

from typing import Dict, Callable, Optional, Tuple
from collections import OrderedDict, defaultdict
from src.apps import ClassificationApp
from src.utils import get_func_from_config
from src.apps.app_utils import update_lr
from src.apps.clients import test
from src.models.model_utils import set_kld_bn_mode, set_weights, copy_pretrain_to_kld

import logging
logger = logging.getLogger(__name__)

class FedL2PClassificationApp(ClassificationApp):    
    def __init__(self, *args, federated_learning=True, 
                    load_model=None, 
                    load_meta_model=None, 
                    skip_initial_eval=False, 
                    skip_central_eval=False,
                    patience=None, **kwargs):
        super().__init__(*args, **kwargs)
        '''
        server weights: List[weight_net_weights]
        '''
        assert 'load_model' is not None

        self.skip_central_eval = skip_central_eval
        if not self.skip_central_eval:
            self.pretrain_net_params = self.ckp.load_model_from_run_id(load_model) 

        logger.info(f'Start from pretrained model from {load_model}')  

        if load_meta_model is not None:
            # download and save meta parameters locally
            logger.info(f'Start from meta parameters from run_id: {load_meta_model}')  
            self.ckp.load_model_from_run_id(load_meta_model, save_as='meta_weights.pkl')

        self.federated_learning = federated_learning
        if not federated_learning:
            logger.info('Standardized learning of personalized policy.')
            self.skip_initial_eval = True
            ## calculate the total number of outer_loop_batches for each client
            app_config = self.ckp.config.app
            server_config = self.ckp.config.server
            data_config = self.ckp.config.data
            budget = app_config.run.num_rounds * server_config.strategy.args.min_fit_clients * app_config.client.args.outer_loop_epochs
            # logger.info(f'{budget}, {app_config.run.num_rounds}, {server_config.strategy.args.min_fit_clients}, {app_config.client.args.outer_loop_batches}')
            budget_per_client = math.ceil(budget / self.ckp.config.simulation.num_clients)
            # assert budget_per_client == int(budget_per_client), f'budget cannot contain decimals: {budget}'
            logger.info(f'{int(budget_per_client)} outer-loop epochs per client.')
            self.budget_per_client = int(budget_per_client)

        else:
            logger.info('Federated learning of personalized policy.')
            self.skip_initial_eval = skip_initial_eval
            if self.skip_initial_eval:
                logger.info(f'Skipping initial evaluation')
            self.budget_per_client = None
            self.patience = int(patience) if patience is not None else patience

    def get_fit_config_fn(self):
        """Return a configuration with static batch size and (local) epochs."""
        def fit_config_fn(rnd: int) -> Dict[str, str]:
            fit_config = self.ckp.config.app.on_fit

            lr_configs = ['inner_lr', 'bn_net_lr', 'mask_net_lr', 'lr_net_lr', 'stop_net_lr']
            lrs = defaultdict(float)

            for lr_config in lr_configs:
                lrs[lr_config] = getattr(fit_config, f'start_{lr_config}')
                if hasattr(fit_config, f'{lr_config}_decay') and getattr(fit_config, f'{lr_config}_decay') is not None:
                    lr_steps = getattr(fit_config, f'{lr_config}_decay').steps
                    lr_factors = getattr(fit_config, f'{lr_config}_decay').factors

                    for step, factor in zip(lr_steps, lr_factors):
                        if rnd > step:
                            lrs[lr_config] *= factor

            self.ckp.log({
                    **lrs
                     }, step=rnd)

            client_config = {
                **lrs,
                "current_round": rnd}
            return client_config

        return fit_config_fn

    def get_evaluate_config_fn(self):
        """"Client evaluate. Evaluate on client's test set"""
        def evaluate_config_fn(rnd: int) -> Dict[str, str]:
            eval_config = self.ckp.config.app.on_evaluate

            eval_epochs = [1]
            if hasattr(eval_config, 'eval_epochs'):
                eval_epochs = eval_config.eval_epochs
            
            full_eval = rnd == self.ckp.config.app.run.num_rounds or not self.federated_learning

            client_config = {
                "standard_learning": not self.federated_learning,
                "budget_train": self.budget_per_client, # standard learning only
                "fit_config": self.get_fit_config_fn()(rnd), # standard learning only
                "lr": eval_config.lr,
                "current_round": rnd,
                "full_eval": full_eval,
                "eval_epochs": eval_epochs, 
                "finetune_epochs": eval_config.finetune_epochs }
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

            # init model. 
            copy_pretrain_to_kld(weights, model)

            set_kld_bn_mode(model, True)
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

        data_config = self.ckp.config.data
        data_class = get_func_from_config(data_config)
        dataset = data_class(self.ckp, **data_config.args)

        group_name = f"{str(dataset.dataset_fl_root).split('/')[-1]}_{self.ckp.config.simulation.num_clients}_{self.ckp.config.app.on_evaluate.finetune_epochs[-1]}"
        if dataset.pre_partition:
            alpha = 0
        else:
            alpha = '_'.join(list(dataset.test_alpha.keys()))

        if not self.skip_central_eval:
            def _centralized_evaluate(rnd, partition, log=True):
                server_metrics = None
                # Evaluate model using strategy implementation (runs eval_fn)
                parameters = self.pretrain_net_params 
                res_cen = server.strategy.evaluate(parameters=parameters, partition=partition)
                if res_cen is not None:
                    loss_cen, server_metrics = res_cen
                    history.add_loss_centralized(rnd=rnd, loss=loss_cen)
                    history.add_metrics_centralized(rnd=rnd, metrics=server_metrics)
                    if log:
                        self.ckp.log(server_metrics, step=rnd)
                return server_metrics

        def _personalized_evaluate(rnd, log=True):
            personalized_metrics = server.evaluate_round(rnd)
            loss_aggregated, metrics_aggregated, _ = personalized_metrics
            
            if log:
                self.ckp.log(metrics_aggregated, step=rnd)
            return metrics_aggregated

        # Initialize parameters to resume training
        if self.load or self.start_run > 1:
            server.parameters = self.current_weights
            logger.info('[*] Global Parameters Loaded.')
        else:
            server.parameters = server._get_initial_parameters()
            # Get initial accuracies
            if not self.skip_initial_eval and self.federated_learning:
                if not self.skip_central_eval:
                    server_metrics = _centralized_evaluate(0, 'test')
                    logger.info(f'Initial Centralized Accuracy: {server_metrics}')
                    for k, v in server_metrics.items():
                        self.ckp.log_summary(k, v)
                        self.ckp.save_results_logfile(group_name, alpha, k, v, ps_type=f'init_{self.ckp.config.name}', reset=False)

                ps_metrics = _personalized_evaluate(0)
                logger.info(f'Initial Personalized Accuracy: {ps_metrics}')

        if self.federated_learning:
            # Run federated learning for num_rounds
            logger.info("FL starting")
            # start_time = timeit.default_timer()

            app_run_config = self.ckp.config.app.run

            lowest_loss = None
            best_rnd = self.start_run
            best_summary = {}
            rnd = self.start_run
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

                    mean_val_loss = np.mean([cm['fed_query_loss'] for cm in clients_metrics])
                    if app_run_config.test_every_n is None and (lowest_loss is None or mean_val_loss < lowest_loss):
                        lowest_loss = mean_val_loss 
                        best_rnd = rnd
                        metrics_aggregated = _personalized_evaluate(rnd)
                        logger.info(f'[Round {rnd}: Lowest Val Loss {round(lowest_loss,2)}] {metrics_aggregated}')
                        self.ckp.save(f'models/best_weights.pkl',
                            server.parameters)
                        best_summary = metrics_aggregated

                    elif app_run_config.test_every_n is not None and rnd % app_run_config.test_every_n == 0:
                        metrics_aggregated = _personalized_evaluate(rnd)
                        logger.info(f'[Round {rnd}: {metrics_aggregated}')

                # end of round saving
                self.ckp.save(f'results/round_{rnd}.pkl', 
                    {'round': rnd,
                    'clients_metrics': clients_metrics, 
                    'server_metrics': server_metrics})
                self.ckp.save(f'models/latest_weights.pkl',
                    server.parameters)
                if rnd == self.start_run or rnd % 25 == 0:
                    # save model every 25 rounds
                    self.ckp.save(f'models/weights_round_{rnd}.pkl',
                        server.parameters)
                self.ckp.save(f'models/last_round_saved.pkl', rnd)

                # check patience
                if self.patience is not None and rnd - best_rnd >= self.patience:
                    logger.info(f'Round {rnd} exceed patience value of {self.patience}. Ending training.')
                    break
                    
            # test set evaluation and logging using wandb summary metrics
            logger.info(f"[Round {rnd}] Training done. Logging best results.")

            for k,v in best_summary.items():
                logger.info(f'Logging {k}:{v}')
                self.ckp.log_summary(k, v)
                self.ckp.save_results_logfile(group_name, alpha, k, v, ps_type=self.ckp.config.name, reset=False)

        else:
            rnd = 0 

            logger.info(f'Running full personalized FL pipeline')
            personalized_metrics = server.evaluate_round(rnd)
            loss_aggregated, metrics_aggregated, _ = personalized_metrics
            
            # save the final tests result
            for k,v in metrics_aggregated.items():
                logger.info(f'Logging {k}:{v}')
                self.ckp.log_summary(k, v)
                self.ckp.save_results_logfile(group_name, alpha, k, v, ps_type=f'nofed_{self.ckp.config.name}', reset=False)

        self.ckp.save(f'models/weights_{rnd}_final.pkl',
                server.parameters)
