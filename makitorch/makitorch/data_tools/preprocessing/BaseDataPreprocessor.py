from makitorch.data_tools.preprocessing.DataPreprocessor import DataPreprocessor

class BaseDataPreprocessor(DataPreprocessor):

    def handle_data(self, images, masks):
        """
            Class which does do any preprocessing        
        """
        return images, masks
