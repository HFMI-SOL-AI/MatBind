import pandas as pd
import torch
from torch.utils.data import Dataset
import numpy as np

class pXRDDataset(Dataset):
    def __init__(self, data: pd.DataFrame, noised: bool = False, scale: float = 0.0025):
        self.data = data["pxrd"].to_list()
        self.noised = noised
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.noised:
            data = self.data[idx] + np.clip(np.random.normal(loc=0, scale=self.scale, size=self.data[idx].shape), 0, 1)
            data = torch.tensor(data).float()
        else:
            data = torch.tensor(self.data[idx]).float()
        return torch.atleast_2d(data)
