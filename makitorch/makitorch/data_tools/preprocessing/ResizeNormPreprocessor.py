from abc import ABC, abstractmethod

import numpy as np
from PIL import Image

from makitorch.data_tools.preprocessing.DataPreprocessor import DataPreprocessor


class ResizeNormPreprocessor(DataPreprocessor):
    MIN_MAX_METHOD = 'minmax'  # to [0, 1]
    STANDARTIZATION_METHOD = 'standartization'  # to mean=0, std=1
    MEAN_METHOD = 'mean'  # to [-1, +1]
    SPECIAL_FUNDUS_METHOD = 'fundus_mask'

    def __init__(self, target_size=None,
                 image_sample=Image.BICUBIC,
                 mask_sample=Image.NEAREST,
                 image_norm=None,
                 mask_norm=None):
        super().__init__()
        
        assert target_size is not None, 'You should specify target image size'
        self.target_size = target_size
        self.image_sample = image_sample
        self.mask_sample = mask_sample
        self.image_norm = image_norm
        self.mask_norm = mask_norm

    def handle_data(self, images, masks):
        """
            Class which does do any preprocessing        
        """
        _images = [image.resize(self.target_size, self.image_sample)
                   for image in images]
        _masks = [mask.resize(self.target_size, self.mask_sample) for mask in masks]

        if self.image_norm is not None:
            _images = getattr(self, self.image_norm)(_images)

        if self.mask_norm is not None:
            _masks = getattr(self, self.mask_norm)(_masks)

        return _images, _masks

    @staticmethod
    def fundus_mask(masks):
        np_masks = [np.array(mask) for mask in masks]
        n_masks = [mask / 255 for mask in np_masks]  #  Norm to [0; 1] considering value 255 as max.
        return [Image.fromarray(mask) for mask in n_masks]

    @staticmethod
    def minmax():
        raise Exception('Unsupported')

    @staticmethod
    def standartization():
        raise Exception('Unsupported')

    @staticmethod
    def mean():
        raise Exception('Unsupported')
