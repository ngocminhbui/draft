from torchvision import datasets, transforms
from base import BaseDataLoader
from .shrec_dataset import ShrecMultiviewDataset
from .shrec_dataset import ArrayDataset

class ArrayDataLoader(BaseDataLoader):
    """Constructs a simple dataloader from array, matrix"""
    def __init__(self, X, Y, batch_size, shuffle, validation_split, num_workers, training=True):
        self.dataset = ArrayDataset(X,Y)
        super(ArrayDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
       
class ShrecDataloader(BaseDataLoader):
    """
    Shrec data loader. Load dataset of views features.
    sample:  (1, n_view, 2048)
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        
        self.data_dir = data_dir
        self.dataset = ShrecMultiviewDataset(data_dir, training)
        
        super(ShrecDataloader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
