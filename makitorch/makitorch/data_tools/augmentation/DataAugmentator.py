from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from PIL.Image import Image

class DataAugmentator(ABC):

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.augment_data(*args, **kwds)

    @abstractmethod
    def augment_data(self, images: List[Image], masks: List[Image]) -> Tuple[List[Image], List[Image]]:
        """
            This method recieves two lists: images and masks on which it applys some data augmentation (e.g. elastic transformation)
        """
        pass
