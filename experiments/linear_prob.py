# script to train a linear probe on frozen representations


# build model
from sklearn.model_selection import train_test_split
import torch
from matbind.model.encoders.crystal_structure.csgcnn import CSGCNN
from matbind.model.encoders.text.bert_encoder import TextEncoder
from matbind.data.datasets.crystal_structure.dataset import CrystalStructureDataset, CrystalGraphData
from matbind.data.datasets.text import TextDataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import RichModelSummary, EarlyStopping
import lightning as L
from torch_geometric.data import Batch
from lightning.pytorch import seed_everything
import torch.nn as nn
from matbind.data.datasets.text import instantiate_bert_tokenizer
from torchmetrics.classification import ConfusionMatrix
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


crystal_system_mapping = {
    "Cubic": 0,
    "Tetragonal": 1,
    "Orthorhombic": 2,
    "Hexagonal": 3,
    "Trigonal": 4,
    "Monoclinic": 5,
    "Triclinic": 6,
}
crystal_system_inverse = {v: k for k, v in crystal_system_mapping.items()}


class LinearProbeModel(torch.nn.Module):
    def __init__(self, input_dim: int, num_classes: int, ckpt_path: str):
        super().__init__()
        self.encoder = TextEncoder(
            config="/p/project1/solai/datasets/matbert-base-cased/config.json",
            checkpoint_path=ckpt_path,
            freeze_encoder=True,
        )
        # self.encoder = CSGCNN(
        #     num_node_features=96,  # do not change this without changing dataloader
        #     num_edge_features=41,
        #     hidden_channels=128,
        #     num_layers=6,
        #     freeze_encoder=True,
        #     ckpt_path=ckpt_path,
        #     remove_head=True,
        #     use_gradient_checkpointing=False,
        # )
        self.linear = nn.Sequential(
            torch.nn.Linear(input_dim, num_classes),
            # torch.nn.ReLU(),
            # torch.nn.Linear(input_dim, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.linear(x)


class CrystalStructureWithBandGapDataset(CrystalStructureDataset):
    def __init__(self, data_frame, property_name: str = "band_gap"):
        super().__init__(data_frame)
        self.property = data_frame[property_name].to_list()

    def __len__(self):
        return len(self.node_features)

    def __getitem__(self, idx):
        property_value = self.property[idx]
        node_features, edge_index, edge_attr = self.node_features[idx], self.edge_index[idx], self.edge_attr[idx]  # type: ignore
        return CrystalGraphData(
            x=torch.from_numpy(np.stack(node_features)),
            edge_index=torch.from_numpy(np.stack(edge_index)),
            edge_attr=torch.from_numpy(np.stack(edge_attr)),
            mid="0",
        ), torch.tensor(property_value, dtype=torch.long)


class TextWithBandGapDataset(TextDataset):
    def __init__(self, data_frame, property_name: str = "band_gap"):
        super().__init__(
            data_frame,
            tokenizer=instantiate_bert_tokenizer(
                pretrained_model_name_or_path="/p/project1/solai/datasets/matbert-base-cased",
                do_lower_case=False,
                tokenizer_parallelism=False,
            ),
        )
        self.property = data_frame[property_name].to_list()

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        property_value = self.property[idx]
        encoding = self.tokenizer(
            self.text_data[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return (
            encoding["input_ids"].squeeze(0),
            encoding["attention_mask"].squeeze(0),
        ), torch.tensor(property_value, dtype=torch.long)


class LinearProbe(L.LightningModule):
    def __init__(self, model: LinearProbeModel, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = torch.nn.CrossEntropyLoss()
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=231)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.long())
        preds = torch.argmax(y_hat, dim=1)
        accuracy = torch.mean((preds == y.long()).float())
        self.log("train_loss", loss, batch_size=y.size(0))
        self.log("train_accuracy", accuracy, batch_size=y.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y.long())
        self.log("val_loss", loss, batch_size=y.size(0))
        # Compute accuracy
        preds = torch.argmax(y_hat, dim=1)
        accuracy = torch.mean((preds == y.long()).float())

        self.log("val_accuracy", accuracy, batch_size=y.size(0))

        # Update confusion matrix
        self.val_confusion_matrix(preds, y.long())

    def on_validation_epoch_end(self):
        """Log confusion matrix at end of validation epoch"""
        cm = self.val_confusion_matrix.compute()

        # Create confusion matrix heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        cm_np = cm.cpu().numpy()

        # Normalize for better visualization
        cm_normalized = cm_np.astype("float") / cm_np.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt=".2f",
            cmap="Blues",
            cbar_kws={"label": "Accuracy"},
            ax=ax,
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion Matrix - Epoch {self.current_epoch}")

        # Log to wandb
        self.logger.log_image(key="confusion_matrix", images=[fig])
        plt.close(fig)

        # Reset confusion matrix for next epoch
        self.val_confusion_matrix.reset()

    def configure_optimizers(self):
        # Only optimize the linear probe, not the frozen encoder
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.learning_rate)


class LinearProbeDataModule(L.LightningDataModule):
    def __init__(self, train_dataset: pd.DataFrame, val_dataset: pd.DataFrame, batch_size: int = 32):
        super().__init__()
        self.train_dataset = TextWithBandGapDataset(train_dataset, "spacegroup")
        self.val_dataset = TextWithBandGapDataset(val_dataset, "spacegroup")
        self.batch_size = batch_size

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn
        )

    @staticmethod
    def collate_fn(batch):
        text_tuple, targets = zip(*batch)
        text_tuple = tuple(torch.stack(items) for items in zip(*text_tuple))
        # graphs, targets = zip(*batch)
        # batched_graph = Batch.from_data_list(graphs)
        targets = torch.stack(targets)
        return text_tuple, targets


def train_linear_probe(
    train_dataset: CrystalStructureWithBandGapDataset,
    val_dataset: CrystalStructureWithBandGapDataset,
    ckpt_path: str | None = None,
):
    data_module = LinearProbeDataModule(train_dataset, val_dataset, batch_size=300)
    model = LinearProbeModel(input_dim=768, num_classes=231, ckpt_path=ckpt_path)
    linear_probe = LinearProbe(model=model, learning_rate=1e-2)
    logger = WandbLogger(project="linear_probe")

    trainer = L.Trainer(
        max_epochs=1000,
        log_every_n_steps=10,
        accelerator="auto",
        devices="auto",
        logger=logger,
        strategy="auto",
        callbacks=[RichModelSummary(max_depth=1), EarlyStopping(monitor="val_loss", patience=5)],
    )
    trainer.fit(linear_probe, datamodule=data_module)


def check_param_difference():
    model_pre = TextEncoder(
        config="/p/project1/solai/datasets/matbert-base-cased/config.json",
        checkpoint_path="/p/project1/solai/datasets/matbert-base-cased",
        freeze_encoder=True,
    )
    print("finished loading pre model")
    model_after = TextEncoder(
        config="/p/project1/solai/datasets/matbert-base-cased/config.json",
        checkpoint_path="/p/project1/solai/yang21/MatBind/experiments/outputs/crystal_structure_dos_pxrd_text/2026-02-09/10-01-52/checkpoints/last.ckpt",
        freeze_encoder=True,
    )
    diff = 0
    for (n1, p1), (n2, p2) in zip(model_pre.named_parameters(), model_after.named_parameters()):
        diff += (p1 - p2).abs().sum().item()

    print("Total diff:", diff)


def crystal_system_distribution(
    data_frame: pd.DataFrame,
    plot_path: str | None = None,
) -> pd.DataFrame:
    """Generate crystal system distribution (counts and percentages) and optionally plot.

    Args:
        data_frame: DataFrame containing a "crystal_system" column.
        plot_path: Optional path to save a distribution plot.

    Returns:
        DataFrame with columns: crystal_system, count, percentage.
    """
    if "crystal_system" not in data_frame.columns:
        raise ValueError("Input DataFrame must contain a 'crystal_system' column.")

    counts = data_frame["crystal_system"].value_counts(dropna=False).sort_index()
    percentages = (counts / counts.sum()) * 100.0
    dist = pd.DataFrame(
        {
            "crystal_system": counts.index,
            "count": counts.values,
            "percentage": percentages.values,
        }
    )

    if plot_path is not None:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=dist,
            x="crystal_system",
            y="count",
            ax=ax,
            color=plt.cm.Blues(0.6),
        )
        ax.set_xlabel("Crystal System")
        ax.set_ylabel("Count")
        ax.set_title("Crystal System Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
    return dist


def space_group_distribution(data_frame: pd.DataFrame,
    plot_path: str | None = None,
) -> pd.DataFrame:
    """Generate crystal system distribution (counts and percentages) and optionally plot.

    Args:
        data_frame: DataFrame containing a "space_group" column.
        plot_path: Optional path to save a distribution plot.

    Returns:
        DataFrame with columns: space_group, count, percentage.
    """
    if "spacegroup" not in data_frame.columns:
        raise ValueError("Input DataFrame must contain a 'space_group' column.")

    counts = data_frame["spacegroup"].value_counts(dropna=False).sort_index()
    print(counts.index)
    percentages = (counts / counts.sum()) * 100.0
    dist = pd.DataFrame(
        {
            "space_group": counts.index,
            "count": counts.values,
            "percentage": percentages.values,
        }
    )

    if plot_path is not None:
        fig, ax = plt.subplots()
        sns.barplot(
            data=dist,
            x="space_group",
            y="count",
            ax=ax,
            color=plt.cm.Blues(0.6),
        )
        ax.set_xlabel("space group")
        ax.set_ylabel("Count")
        ax.set_title("space group Distribution")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
    return dist

if __name__ == "__main__":
    # check_param_difference()
    import pandas as pd
    seed_everything(42)
    # crystal_dataset = pd.read_parquet("/p/project1/solai/datasets/materials_project/crystal_structure.parquet")
    crystal_system = pd.read_csv("/p/project1/solai/datasets/materials_project/symmetry_data.csv")
    text_dataset = pd.read_parquet("/p/project1/solai/datasets/materials_project/text.parquet")
    # dos_dataset = pd.read_parquet("/p/project1/solai/datasets/materials_project/dos.parquet")
    # ckpt_path = "/p/project1/solai/yang21/MatBind/experiments/outputs/crystal_structure_dos_pxrd_text/2026-01-09/18-40-46/checkpoints/epoch_473.ckpt"
    # ckpt_path = "/p/project1/solai/datasets/matbert-base-cased" # original weights
    # ckpt_path = "/p/project1/solai/yang21/MatBind/experiments/outputs/text_scratch/2026-02-04/15-09-35/checkpoints/epoch_389.ckpt" # text scratch
    ckpt_path = "/p/project1/solai/yang21/MatBind/experiments/outputs/crystal_structure_dos_pxrd_text/2026-02-09/10-01-52/checkpoints/last.ckpt" # text as central modality, backbone freeze
    # dataset = pd.read_parquet("/p/project1/solai/datasets/mnumat/mnumat.parquet")
    dataset = text_dataset.merge(crystal_system, on="material_id", how="inner").dropna(subset=["spacegroup"])
    # dataset["crystal_system"] = dataset["crystal_system"].map(crystal_system_mapping)
    # load data
    train_data, val_data = train_test_split(
        dataset,
        train_size=0.95,
        random_state=42,
    )
    train_linear_probe(train_data, val_data, ckpt_path=ckpt_path)
