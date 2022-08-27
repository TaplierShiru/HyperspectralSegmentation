from typing import Tuple
import numpy as np

class DatasetRemapper:

    def __init__(self, old2new: np.ndarray):
        self.old2new = old2new

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        new_mask = mask.copy()
        return image, self.__remap_mask(new_mask, mask)

    def __remap_mask(self, new_mask: np.ndarray, mask: np.ndarray):
        for old_val, new_val in enumerate(self.old2new):
            new_mask[mask == old_val] = new_val
        return new_mask
