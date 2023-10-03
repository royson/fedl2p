import torch
import numpy as np
import os
import shutil
import pickle
import operator
from pathlib import Path
from PIL import Image
from .fl_dataset import FederatedDataset, create_lda_partitions
from .utils import VisionDataset_FL
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

def domainNetTransformation(augment, normalize=False):
    if augment == 'jit':
        return transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
    elif augment:
        transformations = [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation((-30,30)),
                transforms.ToTensor(),
            ]
    else:
        transformations = [
                transforms.ToTensor(),
            ]

    if normalize:
        transformations.append(transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))
    
    return transforms.Compose(transformations)

class DomainNetDataset(FederatedDataset):
    def __init__(self, ckp, *args, val_ratio=0.2, equal_samples=True, normalize=False, **kwargs): 
        super().__init__(ckp, *args, val_ratio=val_ratio, **kwargs)
        self.normalize = normalize

        jit_augment_transformations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-30,30)),
        ]
        jit_normalize_transformations = []

        if normalize:
            jit_augment_transformations.append(transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))
            jit_normalize_transformations.append(transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)))

        self.jit_augment = torch.nn.Sequential(*jit_augment_transformations)
        self.jit_normalize = torch.nn.Sequential(*jit_normalize_transformations)
        
        self.equal_samples = equal_samples
        self.datasets = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']

        self.label_dict = {'bird':0, 
                        'feather':1, 
                        'headphones':2, 
                        'ice_cream':3,
                        'teapot':4, 
                        'tiger':5, 
                        'whale':6, 
                        'windmill':7, 
                        'wine_glass':8, 
                        'zebra':9}     


        self.dataset_fl_root = os.path.join(self.dataset_fl_root, f'domainnet')
        self.fed_train_dir = self.get_fed_dir(self.lda_alpha)
        self.fed_test_dir = self.get_fed_dir(self.test_alpha)
        self.partitions = ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        assert os.path.exists(self.path_to_data), f'DomainNet dataset not found in {self.path_to_data}'
        return

    def _create_fl_partition(self, alpha_dict):
        dir_path = self.get_fed_dir(alpha_dict)
        assert len(alpha_dict) == 1, 'DomainNet experiments only support one alpha group'
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
        num_clients = alpha_dict[alpha]

        assert num_clients % len(self.datasets) == 0, f'[DomainNet Dataset] Num of clients must be divisible by {len(self.datasets)}.'
        dataset_num_clients = num_clients // len(self.datasets)

        partition_ids = [-1 for partition in self.partitions]
        partition_names = [partition.split('.')[0] for partition in self.partitions]
        for pn in partition_names:
            locals()[f'global_x_{pn}'] = None
            locals()[f'global_y_{pn}'] = None

        max_samples = None
        if self.equal_samples:
            # get minimum num of samples
            num_of_samples = []
            for dataset in self.datasets:
                with open(os.path.join(self.path_to_data, f'{dataset}_train.pkl'), 'rb') as f:
                    num_of_samples.append(len(pickle.load(f)[1]))
            max_samples = min(num_of_samples)  

        resize_transform = transforms.Resize([256, 256]) 

        if self.quantity_train_alpha is not None:
            rng = np.random.default_rng(fixed_seed)
            quantity_skew_dirichlet_dist = rng.dirichlet(np.ones(dataset_num_clients) * self.quantity_train_alpha)
        else:
            quantity_skew_dirichlet_dist = None
          
        for dataset in self.datasets:
            dirichlet_dist = None
            dataset_data = {}

            # Loading data from downloaded pickle files
            for partition in ['train', 'test']:
                with open(os.path.join(self.path_to_data, f'{dataset}_{partition}.pkl'), 'rb') as f:
                    filepaths, labels = pickle.load(f)
                
                x = []
                y = []
                for fp, l in zip(filepaths, labels):                
                    image = Image.open(os.path.join(self.path_to_data, '/'.join(fp.split('/')[1:])))
                    if len(image.split()) != 3:
                        image = transforms.Grayscale(num_output_channels=3)(image)
                    image=resize_transform(image)
                    x.append(np.asarray(image)[np.newaxis,:])
                    y.append(self.label_dict[l])
                raw_data = np.concatenate(x, axis=0)
                labels = np.asarray(y)

                if partition == 'train' and self.equal_samples and len(labels) > max_samples:
                    rng = np.random.default_rng(fixed_seed)
                    indices = rng.choice(len(labels), size=max_samples, replace=False)
                    raw_data = np.asarray(operator.itemgetter(*indices)(raw_data))
                    labels = np.asarray(operator.itemgetter(*indices)(labels))
            
                if partition == 'train':
                    # split train and val
                    x_train, x_val, y_train, y_val = train_test_split(raw_data, labels, test_size=self.val_ratio, stratify=labels, shuffle=True, random_state=fixed_seed)
                    for pn in ['train', 'val']:
                        if locals()[f'global_x_{pn}'] is None:
                            locals()[f'global_x_{pn}'] = locals()[f'x_{pn}']
                            locals()[f'global_y_{pn}'] = locals()[f'y_{pn}']
                        else:
                            locals()[f'global_x_{pn}'] = np.concatenate([locals()[f'global_x_{pn}'], locals()[f'x_{pn}']], axis=0)
                            locals()[f'global_y_{pn}'] = np.concatenate([locals()[f'global_y_{pn}'], locals()[f'y_{pn}']], axis=0)

                        dataset_data[pn] = (locals()[f'x_{pn}'], locals()[f'y_{pn}'])
                else:
                    if locals()[f'global_x_test'] is None:
                        locals()[f'global_x_test'] = raw_data
                        locals()[f'global_y_test'] = labels
                    else:
                        locals()[f'global_x_test'] = np.concatenate([locals()[f'global_x_test'], raw_data], axis=0)
                        locals()[f'global_y_test'] = np.concatenate([locals()[f'global_y_test'], labels], axis=0)

                    dataset_data['test'] = (raw_data, labels)

            
            for partition_idx, pn in enumerate(partition_names):  
                idx = partition_ids[partition_idx]

                qs_dirichlet = quantity_skew_dirichlet_dist if 'train' in pn else None                

                client_partitions, dirichlet_dist = create_lda_partitions(
                    dataset=dataset_data[pn],
                    dirichlet_dist=dirichlet_dist,
                    quantity_skew_dirichlet=qs_dirichlet,
                    num_partitions=dataset_num_clients,
                    concentration=float(alpha),
                    accept_imbalanced=True,
                    seed=fixed_seed,
                )

                # saving to disk
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

        if val_ratio:
            assert partition.lower() == 'train'
            assert augment == 'jit'

            train_dataset = VisionDataset_FL(path_to_data=path, 
                transform=domainNetTransformation(augment, self.normalize))
            val_dataset = VisionDataset_FL(path_to_data=os.path.join(prefix_path, f'val.pt'), 
                transform=domainNetTransformation(augment, self.normalize))

            return [DataLoader(train_dataset, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs),
                DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs)]

        else:
            dataset = VisionDataset_FL(path_to_data=path, 
                transform=domainNetTransformation(augment, self.normalize))

            return DataLoader(dataset, batch_size=batch_size, pin_memory=True, drop_last=False, shuffle=shuffle, **kwargs)