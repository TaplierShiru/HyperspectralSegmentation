from makitorch.data_tools.augmentation.DataAugmentator import DataAugmentator

class BaseDataAugmentor(DataAugmentator):

    def augment_data(self, images, masks):
        """
            Class which does do any dasta augmentation        
        """
        return images, masks
