import os
import re
import gc
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as DL
from sklearn.model_selection import train_test_split

import utils as u

#####################################################################################################

class DS(Dataset):
    def __init__(self, X=None, y=None, mode=None, transform=None):
        self.mode = mode
        self.transform = transform
        self.X = X
        if self.mode == "train" or self.mode == "valid":
            self.y = y
    
    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        if self.mode == "train" or self.mode == "valid":
            return self.transform(self.X[idx]), torch.FloatTensor(self.y[idx])
        elif self.mode == "test":
            return self.transform(self.X[idx])
        else:
            raise ValueError("Invalid Argument passed to 'mode'")

#####################################################################################################

def build(path=None, batch_size=None, in_colab=False):
    assert(isinstance(path, str))
    assert(isinstance(batch_size, int))

    u.unzip(path)

    if in_colab:
        m_images_path, f_images_path = os.path.join(u.COLAB_DATA_PATH, "Male"), os.path.join(u.COLAB_DATA_PATH, "Female")
    else:
        m_images_path, f_images_path = os.path.join(u.LOCAL_DATA_PATH, "Male"), os.path.join(u.LOCAL_DATA_PATH, "Female")
    
    m_labels, f_labels = np.zeros((len(os.listdir(m_images_path)), 1)), np.ones((len(os.listdir(f_images_path)), 1))
    m_images, f_images = u.get_images(m_images_path, size=u.PRETRAINED_SIZE), u.get_images(f_images_path, size=u.PRETRAINED_SIZE)
    labels = np.concatenate((m_labels, f_labels), axis=0)
    images = np.concatenate((m_images, f_images), axis=0)

    tr_images, va_images, tr_labels, va_labels = train_test_split(images, labels, test_size=0.2, shuffle=True, random_state=u.SEED)

    tr_data_setup = DS(X=tr_images, y=tr_labels, mode="train", transform=u.FEA_TRANSFORM)
    va_data_setup = DS(X=va_images, y=va_labels, mode="valid", transform=u.FEA_TRANSFORM)

    tr_data = DL(tr_data_setup, batch_size=batch_size, shuffle=True, generator=torch.manual_seed(u.SEED))
    va_data = DL(va_data_setup, batch_size=batch_size, shuffle=False)

    dataloaders = {"train" : tr_data, "valid" : va_data}
    return dataloaders

#####################################################################################################
