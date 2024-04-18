import os
import shutil

import numpy as np
import skimage.io
import torch


def random_split_dataset(input_dataset_path, output_path, ratios=[0.8, 0.1, 0.1],
                         exist_ok=False):
    '''Randomly split a dataset in to the train, validation and test sets.'''
    
    datasets = ['train', 'validation', 'test']
    categories = os.listdir(input_dataset_path)
    ratios = np.array(ratios) / np.sum(ratios)

    for ds in datasets:
        for c in categories:
            os.makedirs(os.path.join(output_path, ds, c), exist_ok=exist_ok)
    
    # Evenly split for each category
    for c in categories:
        files = os.listdir(os.path.join(input_dataset_path, c))
        np.random.shuffle(files)
    
        seps = [0] + [int(r * len(files)) for r in np.cumsum(ratios)]
        
        for i in range(len(datasets)):
            selected_files = files[seps[i] : seps[i + 1]]
            for f in selected_files:
                shutil.copy(os.path.join(input_dataset_path, c, f),
                            os.path.join(output_path, datasets[i], c, f))
    
    

class DiskDataset(torch.utils.data.Dataset):
    '''A dataset on disk.'''
    
    def __init__(self, data_path, normalize_img=False, file_type='tiff'):
        super().__init__()
        self.data_path = data_path
        
        # Define the categories of cell types
        self.categories = os.listdir(data_path)
        
        # Record the files and labels of each cell
        self.files = []
        self.labels = []
        
        for i, c in enumerate(self.categories):
            for f in os.listdir(os.path.join(data_path, c)):
                self.files.append(os.path.join(data_path, c, f))
                self.labels.append(i)
                
        self.files = np.array(self.files)
        self.labels = np.array(self.labels)
        
        # Normalize the image from [0, 255] to [0, 1]
        self.normalize_img = normalize_img

        self.file_type = file_type
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file_subset = self.files[idx]
        label_subset = self.labels[idx]
        
        if type(label_subset) is np.ndarray:
            X_list = []
            for f in file_subset:
                X_list.append(self.load_image(f))
        
            if self.normalize_img:
                return np.array(X_list, dtype=np.float32) / 255, label_subset
            else:
                return np.array(X_list), label_subset
                
        else:
            if self.normalize_img:
                return self.load_image(file_subset).astype(np.float32) / 255, label_subset
            else:
                return self.load_image(file_subset), label_subset

    def load_image(self, file_path):
        if self.file_type == 'tiff':
            return skimage.io.imread(file_path)
        elif self.file_type == 'npz':
            return np.load(file_path)['x']
        else:
            raise Exception(f'Cannot load image file of type {self.file_type}')
