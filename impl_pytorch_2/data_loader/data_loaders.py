from torchvision import datasets, transforms
from base import BaseDataLoader
from .shrec_dataset import ShrecMultiviewDataset
class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super(MnistDataLoader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        
class ShrecDataloader(BaseDataLoader):
    """
    Shrec data loader. Load dataset of views features.
    sample:  (1, n_view, 2048)
    """
    def __init__(self, data_dir, batch_size, shuffle, validation_split, num_workers, training=True):
        
        self.data_dir = data_dir
        self.dataset = ShrecMultiviewDataset(data_dir, training)
        
        super(ShrecDataloader, self).__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
        

if __name__ == "__main__":
    cwd = os.getcwd()

    #test
    shrecdataset = ShrecDataloader('')
    print (shrecdataset[0][0].shape)
    print (shrecdataset[0][0].shape)
    print (len(shrecdataset))
    
