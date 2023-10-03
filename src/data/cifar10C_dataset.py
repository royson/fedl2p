import torch
import numpy as np
import os
import shutil
from pathlib import Path
from .fl_dataset import FederatedDataset, create_lda_partitions
from .utils import VisionDataset_FL
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

def cifar10Transformation(augment):
    if augment == 'jit':
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    elif augment:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

class Cifar10CDataset(FederatedDataset):
    def __init__(self, ckp, corruption_type, severity, *args, split_ratio=0.2, val_ratio=0.2, **kwargs): 
        super().__init__(ckp, *args, val_ratio=val_ratio, **kwargs)
        assert type(severity) == int and (severity >= 0 and severity <= 4) # 4 being most severe
        self.corruption_types = ['brightness', 'frost', 'jpeg_compression', 'shot_noise',
                    'contrast', 'gaussian_blur', 'snow', 'defocus_blur', 'gaussian_noise', 'motion_blur',
                    'spatter', 'elastic_transform', 'glass_blur', 'pixelate', 'speckle_noise', 'fog', 
                    'impulse_noise', 'saturate', 'zoom_blur'] 

        # a subset of corruption types
        self.subset_types = ['brightness', 'frost', 'jpeg_compression',
                    'contrast', 'snow', 'motion_blur',
                    'pixelate', 'speckle_noise', 'fog', 
                    'saturate'] 
        assert corruption_type in self.corruption_types or corruption_type in ['all', 'sub']
        self.jit_augment = torch.nn.Sequential(
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            )
        self.jit_normalize = torch.nn.Sequential(
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            )
        self.split_ratio = split_ratio
        self.corruption_type = corruption_type
        self.dataset_fl_root = os.path.join(self.dataset_fl_root, f'{self.corruption_type}{severity}')
        self.fed_train_dir = self.get_fed_dir(self.lda_alpha)
        self.fed_test_dir = self.get_fed_dir(self.test_alpha)
        self.partitions = ['train.pt', 'val.pt', 'test.pt']
        self.severity = severity
        self.severity_slice = slice(severity * 10000, (severity + 1) * 10000)

    def download(self):
        assert os.path.exists(self.path_to_data), f'CIFAR-10-C dataset not found in {self.path_to_data}'
        return

    def _create_fl_partition(self, alpha_dict):
        dir_path = self.get_fed_dir(alpha_dict)
        assert len(alpha_dict) == 1, 'CIFAR-10-C experiments only support one alpha group'
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

        alpha = next(iter(alpha_dict))

        if self.corruption_type == 'all':
            assert self.pool_size % len(self.corruption_types) == 0, 'num of clients \% number of corruption types must = 0'
            num_clients = self.pool_size // len(self.corruption_types)
            _corruption_types = self.corruption_types
        elif self.corruption_type == 'sub':
            assert self.pool_size % len(self.subset_types) == 0
            num_clients = self.pool_size // len(self.subset_types)
            _corruption_types = self.subset_types
        else:
            _corruption_types = [self.corruption_type]
            num_clients = alpha_dict[alpha]

        partition_ids = [-1 for partition in self.partitions]

        partition_names = [partition.split('.')[0] for partition in self.partitions]

        for pn in partition_names:
            locals()[f'global_x_{pn}'] = None
            locals()[f'global_y_{pn}'] = None

        if self.quantity_train_alpha is not None:
            rng = np.random.default_rng(fixed_seed)
            quantity_skew_dirichlet_dist = rng.dirichlet(np.ones(num_clients) * self.quantity_train_alpha)
        else:
            quantity_skew_dirichlet_dist = None

        for ct in _corruption_types:
            dirichlet_dist = None

            raw_data = np.load(os.path.join(self.path_to_data, f'{ct}.npy'))
            raw_data = raw_data[self.severity_slice]
            labels = np.load(os.path.join(self.path_to_data, f'labels.npy'))
            labels = labels[self.severity_slice]

            # split train and test
            x_train, x_test, y_train, y_test = train_test_split(raw_data, labels, test_size=self.split_ratio, stratify=labels, shuffle=True, random_state=fixed_seed)
            
            # split train and val
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=self.val_ratio, stratify=y_train, shuffle=True, random_state=fixed_seed)

            for partition_idx, pn in enumerate(partition_names):
                if locals()[f'global_x_{pn}'] is None:
                    locals()[f'global_x_{pn}'] = locals()[f'x_{pn}']
                    locals()[f'global_y_{pn}'] = locals()[f'y_{pn}']
                else:
                    locals()[f'global_x_{pn}'] = np.concatenate([locals()[f'global_x_{pn}'], locals()[f'x_{pn}']], axis=0)
                    locals()[f'global_y_{pn}'] = np.concatenate([locals()[f'global_y_{pn}'], locals()[f'y_{pn}']], axis=0)

                group_data = [locals()[f'x_{pn}'], locals()[f'y_{pn}']]
                
                # splitting data into clients
                idx = partition_ids[partition_idx]

                qs_dirichlet = quantity_skew_dirichlet_dist if 'train' in pn else None
                
                client_partitions, dirichlet_dist = create_lda_partitions(
                    dataset=group_data,
                    dirichlet_dist=dirichlet_dist,
                    quantity_skew_dirichlet=qs_dirichlet,
                    num_partitions=num_clients,
                    concentration=float(alpha),
                    accept_imbalanced=True,
                    seed=fixed_seed,
                )

                # saving client_parameters to disk
                for idx, cp in enumerate(client_partitions, idx + 1):
                    client_path = os.path.join(dir_path, str(idx))
                    os.makedirs(client_path, exist_ok=True)
                    torch.save(cp, os.path.join(client_path,f'{pn}.pt'))
                
                partition_ids[partition_idx] = idx

        # saving global datasets
        for pn in partition_names:
            torch.save([locals()[f'global_x_{pn}'], locals()[f'global_y_{pn}']], os.path.join(self.dataset_fl_root, f'{pn}.pt'))

    def get_dataloader(self, 
                    data_pool, 
                    partition,
                    batch_size,
                    num_workers, 
                    augment,
                    cid=None, 
                    path=None,
                    shuffle=False,
                    val_ratio=0,
                    seed=None,
                    **kwargs):
        '''
        Return the class specific dataloader from server or client
        '''
        data_pool = data_pool.lower()
        assert data_pool.lower() in ('server', 'train', 'test'), 'Data pool must be in server, train, or test pool'
        
        if path is not None and os.path.exists(path):
            # forced to use the path 
            # print(f'Forced to use path {path} instead of following data_pool')
            prefix_path = path if cid is None else os.path.join(path, cid)
            path = os.path.join(prefix_path, f'{partition}.pt')
        else:
            if data_pool == 'server':
                assert cid is None
                path = os.path.join(self.dataset_fl_root, f'{partition}.pt')
            elif data_pool == 'train':
                # load training pool of clients
                prefix_path = self.fed_train_dir if cid is None else os.path.join(self.fed_train_dir, cid)
                path = os.path.join(prefix_path, f'{partition}.pt')
            else:
                # load test pool of clients
                prefix_path = self.fed_test_dir if cid is None else os.path.join(self.fed_test_dir, cid)
                path = os.path.join(prefix_path, f'{partition}.pt')

        # print(f'Getting dataloader.. data_pool: {data_pool}, partition: {partition}. \n path: {path}. augment: {augment}. ')

        if val_ratio:
            assert partition.lower() == 'train'
            assert augment == 'jit'

            train_dataset = VisionDataset_FL(path_to_data=path, 
                transform=cifar10Transformation(augment))
            val_dataset = VisionDataset_FL(path_to_data=os.path.join(prefix_path, f'val.pt'), 
                transform=cifar10Transformation(augment))

            return [DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs),
                DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs)]

        else:
            dataset = VisionDataset_FL(path_to_data=path, 
                transform=cifar10Transformation(augment))

            return DataLoader(dataset, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs)