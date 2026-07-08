import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


def pad(max_length, input: torch.Tensor, padding_value: int):
    current_length = input.size(0)

    if current_length >= max_length:
        # No padding needed, just create a mask for the existing length
        padded_tensor = input[:max_length]
        pad_mask = torch.ones(max_length, dtype=torch.long)
        return padded_tensor, pad_mask

    # Calculate the amount of padding needed
    padding_size = max_length - current_length

    # Pad the tensor
    padded_tensor = F.pad(input, pad=(0, padding_size), value=padding_value)

    # Create the padding mask
    pad_mask = torch.cat(
        [
            torch.ones(current_length, dtype=torch.long),
            torch.zeros(padding_size, dtype=torch.long),
        ],
        dim=0,
    )

    return padded_tensor, pad_mask


class DOSDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        noised: bool = False,
        scale: float = 0.0025,
    ):
        self.data = data
        self.max_length = 2001
        self.other_modality = "dos"
        self.energies = self.data["energies"].to_list()
        self.densities = self.data["dos"].to_list()
        self.efermi = self.data["efermi"].to_list()
        self.noised = noised
        self.scale = scale

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        energies = torch.tensor(list(self.energies[idx])) - torch.tensor([self.efermi[idx]])
        energies, energies_mask = pad(self.max_length, energies, 0)

        if self.noised:
            densities = torch.tensor(self.densities[idx] + np.clip(np.random.normal(loc=0, scale=self.scale, size=self.densities[idx].shape), 0, 1))
        else:
            densities = torch.tensor(self.densities[idx])
        densities, densities_mask = pad(self.max_length, densities, 0)

        assert torch.all(energies_mask == densities_mask)
        return densities.unsqueeze(-1).float(), energies.unsqueeze(-1).float(), energies_mask
