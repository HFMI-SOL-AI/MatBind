from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.data import Dataset as GeometricDataset


@dataclass
class CrystalGraphData(Data):
    x: torch.Tensor

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        mid: str,
    ):
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mid=mid,
        )


class CrystalStructureDataset(GeometricDataset):
    def __init__(self, data: pd.DataFrame) -> None:
        required_columns = ["node_features", "edge_index", "edge_attr"]
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"The dataset does not contain the required columns: {required_columns}")

        self.node_features = data["node_features"].to_list()
        self.edge_index = data["edge_index"].to_list()
        self.edge_attr = data["edge_attr"].to_list()
        self.mids = data["material_id"].to_list()

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        node_features, edge_index, edge_attr = self.node_features[idx], self.edge_index[idx], self.edge_attr[idx]  # type: ignore
        return CrystalGraphData(
            x=torch.from_numpy(np.stack(node_features)),
            edge_index=torch.from_numpy(np.stack(edge_index)),
            edge_attr=torch.from_numpy(np.stack(edge_attr)),
            mid = self.mids[idx],
        )
