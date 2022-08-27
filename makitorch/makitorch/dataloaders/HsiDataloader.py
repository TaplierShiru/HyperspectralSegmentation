from makitorch.data_tools.augmentation import DataAugmentator
from makitorch.data_tools.augmentation import BaseDataAugmentor
from makitorch.data_tools.preprocessing import BaseDataPreprocessor
from makitorch.data_tools.preprocessing import DataPreprocessor

from typing import Callable, Optional, Union

import torch
from sklearn.utils import shuffle
from hsi_dataset_api import HsiDataset


class HsiDataloader(torch.utils.data.IterableDataset):
    def __init__(
            self, 
            data_path: str,
            preprocessing: Optional[Union[DataPreprocessor, Callable]] = BaseDataPreprocessor(),
            augmentation: Optional[Union[DataAugmentator, Callable]] = BaseDataAugmentor(),
            indices = None,
            shuffle_data=False
        ):
        super().__init__()
        self.shuffle_data = shuffle_data
        self.dataset = HsiDataset(data_path)
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        
        self.images = []
        self.masks = []
        
        for idx, data_point in enumerate(self.dataset.data_iterator(opened=True, shuffle=False)):
            if indices is not None and idx not in indices:
                continue
            self.images.append(data_point.hsi)
            self.masks.append(data_point.mask)
        
        if self.preprocessing is not None:
            self.images, self.masks = self.preprocessing(self.images, self.masks)
        
    def __iter__(self):
        if self.shuffle_data:
            self.images, self.masks = shuffle(self.images, self.masks)
        
        for image, mask in zip(self.images, self.masks):
            yield self.augmentation(image, mask)
