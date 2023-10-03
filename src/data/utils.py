import torch
import os
import numpy as np
from pathlib import Path
from PIL import Image

from typing import Dict, Callable, List, Optional, Tuple, Any
from torchvision.datasets import VisionDataset

class VisionDataset_FL(VisionDataset):
    def __init__(
        self,
        path_to_data=None,
        images=None,
        targets=None, 
        transform: Optional[Callable] = None,
    ) -> None:
        path = Path(path_to_data).parent if path_to_data else None
        assert path_to_data is not None or (images is not None and targets is not None)
        super(VisionDataset_FL, self).__init__(path, transform=transform)
        self.transform = transform
        
        if path_to_data:
            self.data, self.targets = torch.load(path_to_data)
        else:
            self.data = images
            self.targets = targets

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if not isinstance(img, Image.Image):  
            if not isinstance(img, np.ndarray):  
                img = img.numpy()

            img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
