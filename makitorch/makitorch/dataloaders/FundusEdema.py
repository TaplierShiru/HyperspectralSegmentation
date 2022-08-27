from makitorch.data_tools.augmentation import DataAugmentator
from makitorch.data_tools.augmentation import BaseDataAugmentor
from makitorch.data_tools.preprocessing import BaseDataPreprocessor
from makitorch.data_tools.preprocessing import DataPreprocessor

from typing import Callable, List, Optional, Union
import os
from glob import glob

import torch
import PIL
import cv2
import numpy as np


class FundusDataloder(torch.utils.data.IterableDataset):
    def __init__(self,
                 data_path: str,
                 image_prefix: str,
                 mask_prefix: str,
                 preprocessing: Optional[Union[DataPreprocessor, Callable]] = BaseDataPreprocessor(),
                 augmentation: Optional[Union[DataAugmentator, Callable]] = BaseDataAugmentor(),
                 sigmaX: int = 30,
                 **kwargs):
        super().__init__()
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        images_path = sorted(glob(os.path.join(data_path, image_prefix)))
        masks_path = sorted(glob(os.path.join(data_path, mask_prefix)))

        self.images = [PIL.Image.open(image_path)
                       for image_path in images_path]
        self.masks = [PIL.Image.open(mask_path).convert('L')
                      for mask_path in masks_path]
        assert len(self.images) == len(
            self.masks), 'Length of images and masks are not equal'
        
        # self.images = [PIL.Image.fromarray(self._preprocessing_image(np.array(img), sigmaX)) for img in self.images]
        # self.masks = [PIL.Image.fromarray(self._preprocessing_mask(np.array(mask))) for mask in self.masks]
        if preprocessing is not None:
            self.images, self.masks = preprocessing(self.images, self.masks, **kwargs)

    def __iter__(self):
        for idx in range(len(self.images)):
            image = self.images[idx]
            mask = self.masks[idx]
            yield self.augmentation(image, mask)

    def _preprocessing_image(self, image: np.ndarray, sigmaX: int):
        return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    def _preprocessing_mask(self, mask):
        mask[mask != 0] = mask[mask != 0] - np.min(mask[mask != 0]) + 1
        return mask


class ComplexFundusDataloder(torch.utils.data.IterableDataset):
    def __init__(self,
                 data_path: str,
                 image_prefix: str,
                 mask_prefix: str,
                 bmask_prefix: str,
                 preprocessing: Optional[Union[DataPreprocessor, Callable]] = BaseDataPreprocessor(),
                 augmentation: Optional[Union[DataAugmentator, Callable]] = BaseDataAugmentor(),
                 sigmaX: int = 30,
                 **kwargs):
        super().__init__()
        self.augmentation = augmentation
        self.preprocessing = preprocessing

        images_path = sorted(glob(os.path.join(data_path, image_prefix)))
        masks_path = sorted(glob(os.path.join(data_path, mask_prefix)))
        bmasks_path = sorted(glob(os.path.join(data_path, bmask_prefix)))

        self.images = [PIL.Image.open(image_path)
                       for image_path in images_path]
        self.masks = [PIL.Image.open(mask_path).convert('L')
                      for mask_path in masks_path]
        self.bmasks = [PIL.Image.open(mask_path).convert('L')
                      for mask_path in bmasks_path]
        
        self.files_id = [os.path.basename(image_path).split('.')[0] for image_path in images_path]

        assert len(self.images) == len(self.masks), \
              'Length of images and masks are not equal'
        
        self.images = [PIL.Image.fromarray(self._preprocessing_image(np.array(img), sigmaX)) for img in self.images]
        if preprocessing is not None:
            self.images, self.masks, self.bmasks = preprocessing(self.images, self.masks, self.bmasks)

    def __iter__(self):
        for idx in range(len(self.images)):
            file_id = self.files_id[idx]
            image = self.images[idx]
            mask = self.masks[idx]
            b_mask = self.bmasks[idx]
            yield file_id, *self.augmentation(image, mask, b_mask)

    def _preprocessing_image(self, image: np.ndarray, sigmaX: int):
        return cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)

    def _preprocessing_mask(self, mask):
        mask[mask != 0] = mask[mask != 0] - np.min(mask[mask != 0]) + 1
        return mask