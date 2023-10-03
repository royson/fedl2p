import numpy as np
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from .fl_dataset import FederatedDataset

from .utils import VisionDataset_FL
from sklearn.model_selection import train_test_split
import pickle
from typing import List
from PIL import Image

from .speech_commands import get_speechcommands_and_partition_it, PartitionedSPEECHCOMMANDS, raw_audio_to_AST_spectrogram

import logging
logger = logging.getLogger(__name__)


class CommandsDataset(FederatedDataset):
    def __init__(self, *args, unseen=False, version=2, max_train=250, max_unseen=50, classes=12, **kwargs): 
        super().__init__(*args, **kwargs)
        self.max_train = max_train
        self.max_unseen = max_unseen
        self.classes = classes
        self.unseen = unseen
        self.version = version

        ## all transformation is handled in speech_commands
        self.jit_augment = torch.nn.Sequential(*[])
        self.jit_normalize = torch.nn.Sequential(*[])

        self.download() # overwrites self.dataset_fl_root
        
    def download(self):
        self.dataset_fl_root = get_speechcommands_and_partition_it(self.path_to_data, version=self.version, max_train=self.max_train, max_unseen=self.max_unseen)
        self.dataset_fl_root = self.dataset_fl_root / 'unseen' if self.unseen else self.dataset_fl_root / 'training'

    def get_available_training_clients(self):
        return list(range(self.max_train)) if not self.unseen else list(range(self.max_unseen))

    def get_dataloader(self, 
                    data_pool, 
                    partition,
                    batch_size,
                    num_workers, 
                    augment,
                    shuffle=False,
                    cid=None, 
                    path=None,
                    val_ratio=False, 
                    seed=None,
                    **kwargs):
        '''
        Return the class specific dataloader from server or client
        '''
        data_pool = data_pool.lower()
        assert data_pool.lower() in ('server', 'train', 'test'), 'Data pool must be in server, train, or test pool'
        assert partition.lower() in ('train', 'test')

        if path is not None:
            assert os.path.exists(path)
            # print(f'Forced to use path {path} instead of following data_pool')
            path = path if cid is None else os.path.join(path, cid)
            path = Path(path)
        else:
            if data_pool == 'server':
                assert cid is None
                path = self.dataset_fl_root
            else:
                assert cid is not None
                path = self.dataset_fl_root / cid

        # print(f'Getting dataloader.. data_pool: {data_pool}, partition: {partition}. \n path: {path}. val_ratio: {val_ratio}, augment: {augment}. ')

        if partition.lower() == 'train':
            dataset = PartitionedSPEECHCOMMANDS(path, "training", transforms=raw_audio_to_AST_spectrogram(), wav2fbank=True, classes=self.classes)
            sampler, _ = dataset.get_balanced_sampler()
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset._collate_fn, sampler=sampler, pin_memory=True, **kwargs)
        else: #test
            dataset = PartitionedSPEECHCOMMANDS(path, "testing", transforms=raw_audio_to_AST_spectrogram(), wav2fbank=True, classes=self.classes)
            dataloader = DataLoader(dataset, batch_size=batch_size, pin_memory=True, collate_fn=dataset._collate_fn, shuffle=shuffle, drop_last=False, **kwargs)

        if val_ratio:
            # loads val together with train
            assert partition.lower() == 'train'
            val_dataset = PartitionedSPEECHCOMMANDS(path, "validation", transforms=raw_audio_to_AST_spectrogram(), wav2fbank=True, classes=self.classes)
            return [dataloader, DataLoader(val_dataset, batch_size=batch_size, pin_memory=True, collate_fn=dataset._collate_fn, shuffle=shuffle, drop_last=False, **kwargs)] 
        else:
            return dataloader