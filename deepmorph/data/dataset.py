import os
import numpy as np
import skimage.io
import torch

class DiskDataset(torch.utils.data.Dataset):
    '''A dataset on disk.'''
    
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        
        # Define the categories of cell types
        self.categories = os.listdir(data_path)
        
        # Record the files and labels of each cell
        self.files = []
        self.labels = []
        
        for c in self.categories:
            for f in os.listdir(os.path.join(data_path, c)):
                self.files.append(os.path.join(data_path, c, f))
                self.labels.append(c)
                
        self.files = np.array(self.files)
        self.labels = np.array(self.labels)
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_subset = self.files[idx]
        label_subset = self.labels[idx]
        
        if type(label_subset) is np.ndarray:
            X_list = []
            for f in file_subset:
                X_list.append(skimage.io.imread(f))
        
            return np.array(X_list), label_subset
        
        else:
            return skimage.io.imread(file_subset), label_subset