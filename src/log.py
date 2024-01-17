import os
import sys
import wandb
import yaml
import copy
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from datetime import datetime
from config import AttrDict
from filelock import FileLock, Timeout
from pathlib import Path
from collections import defaultdict


import logging
logger = logging.getLogger(__name__)

class Checkpoint():
    def __init__(self, args): 
        if 'name' not in args:
            args['name'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self.name = args['name']
        self._config = None

        if 'run_id' not in args:
            args['run_id'] = wandb.util.generate_id()
    
        self.init_logger(args['debug'])
        self.args = args

    @property
    def config(self):
        if self._config is None:
            logger.warn('wandb config has not been initialized. Call Checkpoint.init_wandb()')
        return self._config

    def update_wandb_config(self):
        # Call this to update wandb config
        wandb.config.update(self._config, allow_val_change=True)
    
    def init_wandb(self):
        program_args = copy.deepcopy(self.args)
        dont_save = ['debug', 'run']
        for a in dont_save:
            program_args.pop(a, None)

        wandb_args = self.args['wandb_args']
        
        # Initialize wandb
        os.environ["WANDB_SILENT"] = "false"
        wandb.init(
            id=self.args['run_id'], 
            name=self.name,
            config=program_args,
            resume='allow',
            **wandb_args
        )
        os.environ["WANDB_SILENT"] = "true"
        self._config = AttrDict(wandb.config)
        self.run_dir = wandb.run.dir

        # wandb saves yaml in a different format than how we like to process yaml
        # we save the arguments of this run to a different file for us to easily resume run
        if not os.path.exists(os.path.join(self.run_dir, 'user_config.yaml')):
            with open(os.path.join(self.run_dir, 'user_config.yaml'), 'w') as f:
                logger.info(f"Saving user\'s config to {self.run_dir}")
                yaml.dump(program_args, f)
        
        # Save run ID
        if self.args['save_run_id']:
            os.makedirs(self.args['save_id_path'], exist_ok=True)
            pkl_path = os.path.join(self.args['save_id_path'], 'run_ids.pkl')
            d = {}
            if os.path.exists(pkl_path):
                lock = FileLock("{}.lock".format(pkl_path))
                try:
                    with lock.acquire(timeout = 10):
                        with open(pkl_path, 'rb') as handle:
                            d = pickle.load(handle)
                except Timeout:
                    logger.error('Lock not acquired to read pickle file for run_ids.')
            if self.name in d:
                d[self.name].append(self.args['run_id'])
            else:
                d[self.name] = [self.args['run_id']]
            lock = FileLock("{}.lock".format(pkl_path))
            try:
                with lock.acquire(timeout = 10):
                    with open(pkl_path, 'wb') as handle:
                        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except Timeout:
                logger.error('Lock not acquired to update pickle file for run_ids.')

        

    def init_logger(self, debug=False):
        # Initializing logging
        handlers = []
        handlers.append(logging.StreamHandler(sys.stdout)) 
        
        level = logging.DEBUG if debug else logging.INFO
        
        logging.basicConfig(
            level=level,
            handlers=handlers)
        logging.debug(f'[*] Logging debug mode.')

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)

    def log_summary(self, k, v):
        # useful for storing a single value that summarizes the training
        wandb.run.summary[k] = v

    def plot(self, title, x, y, plt_type='plot', ylabel='Weight', xlabel='Loss', figsize=(20, 5), dpi=80):
        figure(figsize=figsize, dpi=dpi)
        getattr(plt, plt_type)(x,y)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(os.path.join(self.run_dir, f'{title}.png'))
        plt.clf()

    def get_weights_distributions(self, model):
        ''' 
        For all parameters in conv and linear layers,
        generate an N-bin (max 512) histogram to later log to W&B. 
        '''
        w_hists = {}
        bins = 256
        for name, param in model.named_parameters():

            # if weights of a 3x3 convolutional layer
            if len(param.shape) == 4:  # if a conv layer
                if param.shape[2] == 3 and param.shape[3] == 3:  # if 3x3 conv
                    w_hists[name] = wandb.Histogram(param.detach().cpu(),
                                                    num_bins=bins)

        return w_hists

    def save(self, filepath, value, sync_to_cloud=True):
        '''
        Save user-defined values in wandb dir. 
        '''
        # create parent directories if they don't exist
        if os.path.dirname(filepath): 
            os.makedirs(os.path.join(self.run_dir, os.path.dirname(filepath)), exist_ok=True)

        with open(os.path.join(self.run_dir, filepath), 'wb') as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

        policy = 'live' if sync_to_cloud else 'end'
        wandb.save(filepath, policy=policy)
    
    def offline_save(self, filepath, value):
        '''
        Save to local filepath only. Does not upload to wandb instead a sync is initialized.
        '''
        # create parent directories if they don't exist
        if os.path.dirname(filepath): 
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'wb') as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        '''
        Load user-defined values in wandb dir by 
        1) restoring the file from cloud to the new local run and then
        2) loading from that file
        '''
        try:
            filepath = wandb.restore(filepath).name
        except (ValueError, AttributeError):
            logger.info(f'Trying to restore {filepath} but not found. Trying local path..')
            return self.offline_load(os.path.join(self.run_dir, filepath))
            # return None
            
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    def offline_load(self, filepath):
        '''
        Load from local filepath that hasn't been uploaded to wandb
        '''
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            logger.info(f'{filepath} not found.')
            return None

    def load_model_from_run_id(self, wandb_run_path, 
        save_as='latest_weights.pkl'):
        '''
        Load weights from wandb_run_path which consists of 'entity/project/run_id/{path}'
        '''
        parameters = None
        pretrain_path = f'{self.run_dir}/pretrained_model'
        pretrain_local_path = os.path.join(pretrain_path, save_as)

        if not os.path.exists(pretrain_local_path):
            api = wandb.Api()
            split_path = Path(wandb_run_path).parts
            assert len(split_path) >= 3, 'invalid wandb run path'
            wandb_run = os.path.join(*split_path[:3])
            wandb_folder = split_path[3:]

            if not wandb_folder:
                # backward compatibility & default value
                wandb_folder = 'models/latest_weights.pkl'
            else:
                wandb_folder = os.path.join(*wandb_folder)

            try:
                run = api.run(wandb_run)
            except:
                logger.error(f'{wandb_run} not found')
            if run.state == 'crashed' or run.state == 'failed':
                logger.error(f'{wandb_run} crashed/failed')
                            
            run.file(wandb_folder).download(replace=False, root=pretrain_path)
            os.rename(os.path.join(pretrain_path, wandb_folder), os.path.join(pretrain_path, save_as))
            try:
                wandb.save(f'pretrained_model/{save_as}')
            except:
                logger.warning(f'wandb has not been initialized. cannot sync pretrain model.')
        with open(pretrain_local_path, 'rb') as f:
            parameters = pickle.load(f)
        return parameters

    def save_results_logfile(self, group_name, alpha, k, v, filepath='results.pkl', ps_type='default', reset=False):
        '''
        Parse results into a dictionary which is then saved in a pickle file 
        '''
        if not os.path.exists(filepath):
            logger.info(f'Results {filepath} not found! Creating new..')
            all_results = defaultdict(dict)

        logger.info(f'Saving to all results logfile: {filepath}')
        lock = FileLock("{}.lock".format(filepath))
        try:
            with lock.acquire(timeout = 60):
                if os.path.exists(filepath):
                    with open(filepath, 'rb') as handle:
                        all_results = pickle.load(handle)
                
                if group_name not in all_results:
                    all_results[group_name] = defaultdict(dict)
                if alpha not in all_results[group_name]:
                    all_results[group_name][alpha] = defaultdict(dict)
                if ps_type not in all_results[group_name][alpha]:
                    all_results[group_name][alpha][ps_type] = defaultdict(list)
                    
                if reset:
                    all_results[group_name][alpha][ps_type][k] = [v]                    
                else:
                    all_results[group_name][alpha][ps_type][k].append(v)
            
                with open(filepath, 'wb') as handle:
                    pickle.dump(all_results, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
        except Timeout:
            logger.error('Lock not acquired to read pickle file.')
            

