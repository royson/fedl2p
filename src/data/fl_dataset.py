import torch
import shutil
import os
import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split
from flwr.dataset.utils.common import shuffle, sort_by_label, split_array_at_indices, sample_without_replacement

import logging
logger = logging.getLogger(__name__)

## from flwr 0.19.0. seed argument is missing in version 0.17.0
def create_lda_partitions(
    dataset,
    dirichlet_dist,
    quantity_skew_dirichlet=None,
    num_partitions= 100,
    concentration=0.5,
    accept_imbalanced = False,
    seed= None,
):
    x, y = dataset
    x, y = shuffle(x, y)
    x, y = sort_by_label(x, y)

    if (x.shape[0] % num_partitions) and (not accept_imbalanced):
        raise ValueError(
            """Total number of samples must be a multiple of `num_partitions`.
               If imbalanced classes are allowed, set
               `accept_imbalanced=True`."""
        )

    num_samples = num_partitions * [0]
    if quantity_skew_dirichlet is None:
        for j in range(x.shape[0]):
            num_samples[j % num_partitions] += 1
    else:
        # must have at least 2 samples
        num_samples = np.random.multinomial(x.shape[0] - (2 * num_partitions), quantity_skew_dirichlet)
        num_samples += 2
        assert np.sum(num_samples) == x.shape[0]

    # Get number of classes and verify if they matching with
    classes, start_indices = np.unique(y, return_index=True)

    # Make sure that concentration is np.array and
    # check if concentration is appropriate
    concentration = np.asarray(concentration)

    # Check if concentration is Inf, if so create uniform partitions
    partitions: List[XY] = [(_, _) for _ in range(num_partitions)]
    if float("inf") in concentration:

        partitions = create_partitions(
            unpartitioned_dataset=(x, y),
            iid_fraction=1.0,
            num_partitions=num_partitions,
        )
        dirichlet_dist = get_partitions_distributions(partitions)[0]

        return partitions, dirichlet_dist

    if concentration.size == 1:
        concentration = np.repeat(concentration, classes.size)
    elif concentration.size != classes.size:  # Sequence
        raise ValueError(
            f"The size of the provided concentration ({concentration.size}) ",
            f"must be either 1 or equal number of classes {classes.size})",
        )

    # Split into list of list of samples per class
    list_samples_per_class: List[List[np.ndarray]] = split_array_at_indices(
        x, start_indices
    )

    if dirichlet_dist is None:
        dirichlet_dist = np.random.default_rng(seed).dirichlet(
            alpha=concentration, size=num_partitions
        )

    if dirichlet_dist.size != 0:
        if dirichlet_dist.shape != (num_partitions, classes.size):
            raise ValueError(
                f"""The shape of the provided dirichlet distribution
                 ({dirichlet_dist.shape}) must match the provided number
                  of partitions and classes ({num_partitions},{classes.size})"""
            )

    # Assuming balanced distribution
    empty_classes = classes.size * [False]
    for partition_id in range(num_partitions):
        partitions[partition_id], empty_classes = sample_without_replacement(
            distribution=dirichlet_dist[partition_id].copy(),
            list_samples=list_samples_per_class,
            num_samples=num_samples[partition_id],
            empty_classes=empty_classes,
        )

    return partitions, dirichlet_dist

class FederatedDataset(ABC):
    def __init__(self, ckp, path_to_data, dataset_fl_root, *args, 
        pre_partition=False,
        lda_alpha=None,
        val_ratio=0, 
        train_alpha=None, 
        test_alpha=None, 
        quantity_train_alpha=None,
        reset=False, **kwargs):
        if hasattr(ckp.config, 'seed'):
            np.random.seed(ckp.config.seed)

        self.ckp = ckp
        self.config = ckp.config
        self.pool_size = ckp.config.simulation.num_clients
        self.pre_partition = pre_partition
        self.path_to_data = path_to_data
        self.dataset_fl_root = dataset_fl_root
        self.lda_alpha = lda_alpha
        self.train_alpha = train_alpha
        self.test_alpha = test_alpha 
        self.quantity_train_alpha = quantity_train_alpha       
        self.val_ratio = val_ratio
        self.reset = reset
        self.partitions = ['train.pt', 'test.pt']

        if not self.pre_partition:
            assert self.lda_alpha is not None, 'dataset is not pre-partitioned. data.args.lda_alpha must be defined.'
            # sort lda_alpha as loading from saved dict might violate the order
            self.lda_alpha = dict(sorted(self.lda_alpha.items(), reverse=True))
            self.lda_alpha = {str(key): value for key, value in self.lda_alpha.items()}

            if self.train_alpha is None:
                self.train_alpha = list(self.lda_alpha.keys())
            self.train_alpha = list(map(lambda x: str(x), self.train_alpha))
            self.train_alpha = sorted(self.train_alpha, reverse=True)
            if self.test_alpha is None:
                self.test_alpha = self.lda_alpha
            self.test_alpha = dict(sorted(self.test_alpha.items(), reverse=True))

            for a in self.train_alpha:
                assert str(float(a)) in self.lda_alpha or str(int(a)) in self.lda_alpha, f'Train alpha ({a}) must be found in lda_alpha ({self.lda_alpha})'
            assert self.pool_size == sum(self.lda_alpha.values()) == sum(self.test_alpha.values()), \
                'Num of clients must match total no. of clients defined in data.args.lda_alpha and data.args.test_alpha'

            self.fed_train_dir = self.get_fed_dir(self.lda_alpha)
            self.fed_test_dir = self.get_fed_dir(self.test_alpha)

    def create_fl_partitions(self):
        assert not self.pre_partition, 'FL dataset is pre-partitioned. Do not recreate fl partitions'
        self._create_fl_partition(self.lda_alpha)
        self._create_fl_partition(self.test_alpha)
        logger.info(f'Lda alpha:{self.lda_alpha}. Test alpha: {self.test_alpha}. Training with {len(self.get_available_training_clients())} clients.')

    def get_fed_dir(self, alpha_dict):
        name = ''
        for alpha, num_clients in alpha_dict.items():
            name += f'{alpha}_{num_clients}_'
        if self.quantity_train_alpha is not None:
            name += f'QT{self.quantity_train_alpha}_'
        name += f'valratio{self.val_ratio}'

        return os.path.join(self.dataset_fl_root, str(self.pool_size), name)

    def get_available_training_clients(self):
        if not self.pre_partition:
            available_clients = []
            start = 0
            for alpha, num_clients in self.lda_alpha.items():
                if alpha in self.train_alpha:
                    available_clients += list(range(start, start + num_clients))
                start += num_clients

            return available_clients
        else:
            raise NotImplementedError # overwrite this method if pre-partitioned

    def _create_fl_partition(self, alpha_dict):
        dir_path = self.get_fed_dir(alpha_dict)
        os.umask(0)

        if self.reset and os.path.exists(dir_path):
            logger.info(f'Reset flag is set for data federated splitting.. Deleting current {dir_path}')
            shutil.rmtree(dir_path)

        if self._has_fl_partition(dir_path):
            logger.info(f"FL partitioned dataset {dir_path} found.")
            return 
        
        self.download()
        
        logger.info(f"Creating FL partitioned dataset {dir_path}..")

        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path)

        fixed_seed = self.config.seed if hasattr(self.config, 'seed') else 42

        dirichlet_dists = {}
        quantity_skew_dirichlet_dists = {}
        for alpha in alpha_dict.keys():
            dirichlet_dists[alpha] = None
            if self.quantity_train_alpha is not None:
                rng = np.random.default_rng(fixed_seed)
                quantity_skew_dirichlet_dists[alpha] = rng.dirichlet(np.ones(alpha_dict[alpha]) * self.quantity_train_alpha)
            else:
                quantity_skew_dirichlet_dists[alpha] = None

        for partition in self.partitions:
            alpha_data = {}
            raw_data_dir = os.path.join(self.path_to_data, f'{partition}')
            server_data_dir = os.path.join(self.dataset_fl_root, f'{partition}')
            if not os.path.exists(server_data_dir):
                shutil.copyfile(raw_data_dir, server_data_dir)
            X_raw, Y_raw = torch.load(raw_data_dir)
            if partition == 'train.pt' and self.val_ratio > 0.:
                # split train and split
                val_path = os.path.join(dir_path, "val.pt")
                X_raw, X_val, Y_raw, y_val = train_test_split(
                    X_raw, Y_raw, test_size=self.val_ratio, stratify=Y_raw, shuffle=True, random_state=fixed_seed
                )
                torch.save([X_val, y_val], val_path)

            # splitting partition into groups
            unallocated_size = self.pool_size
            for alpha, num_clients in alpha_dict.items():
                ratio = num_clients / unallocated_size
                if ratio == 1:
                    X_group = X_raw
                    Y_group = Y_raw
                else:
                    X_raw, X_group, Y_raw, Y_group = train_test_split(
                        X_raw, Y_raw, test_size=ratio, stratify=Y_raw, random_state=fixed_seed)
                alpha_data[alpha] = (X_group, Y_group)
                unallocated_size -= num_clients
            
            # splitting each group into each client
            idx = -1
            for alpha, group_data in alpha_data.items():
                num_clients = alpha_dict[alpha]

                qs_dirichlet = quantity_skew_dirichlet_dists[alpha] if 'train' in partition else None 
                
                client_partitions, dirichlet_dist = create_lda_partitions(
                    dataset=group_data,
                    dirichlet_dist=dirichlet_dists[alpha],
                    quantity_skew_dirichlet=qs_dirichlet,
                    num_partitions=num_clients,
                    concentration=float(alpha),
                    accept_imbalanced=True,
                    seed=fixed_seed,
                )
                dirichlet_dists[alpha] = dirichlet_dist

                # saving client_parameters to disk
                for idx, cp in enumerate(client_partitions, idx + 1):
                    client_path = os.path.join(dir_path, str(idx))
                    os.makedirs(client_path, exist_ok=True)
                    torch.save(cp, os.path.join(client_path,partition))

        
    def _has_fl_partition(self, dir_path):
        if self.val_ratio > 0 and not os.path.exists(os.path.join(self.dataset_fl_root, 'val.pt')):
            return False
        
        for cid in range(self.pool_size):
            for partition in self.partitions:
                file_path = os.path.join(dir_path, str(cid), partition)
                if not os.path.exists(file_path):
                    return False    

        return True

    @abstractmethod
    def download(self):
        '''
        Downloads the dataset to self.path_to_data
        '''

    @abstractmethod
    def get_dataloader(self, 
                    data_pool, # server/train/test
                    partition,
                    *args,
                    cid=None, 
                    **kwargs):
        '''
        Class-specific dataloader
        '''
        




