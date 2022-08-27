from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from PIL.Image import Image

class DataPreprocessor(ABC):

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.handle_data(*args, **kwds)

    @abstractmethod
    def handle_data(self, images: List[Image], masks: List[Image]) -> Tuple[List[Image], List[Image]]:
        """
            This method recieves two lists: images and masks on which it applys some data preprocessing (e.g. resize)
        """
        pass
