import datetime
from pathlib import Path
import pickle as pkl
from pprint import pformat
import hydra
import lightning.pytorch as L
from lightning_fabric import seed_everything
import pandas as pd
import rootutils
from dotenv import load_dotenv
from omegaconf import DictConfig

from matbind.data.analysis import aggregate_embeddings
from matbind.data.datamodules.matbind import MatBindDataModule
from matbind.metrics.retrieval import full_database_retrieval
from matbind.model.lightning_module import MatBindModule
from matbind.utils.pylogger import get_pylogger
import torch.nn as nn

LOGGER = get_pylogger(__name__)

load_dotenv()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# set a unique identifier for the retrieval run
RETRIEVAL_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M")

def embed(config: DictConfig):
    LOGGER.info("Starting data module and model instantiation...")
    datamodule: MatBindDataModule = hydra.utils.instantiate(config.data)
    # datamodule.setup()

    LOGGER.info("Starting model instantiation...")
    trainer: L.Trainer = hydra.utils.instantiate(
        config.trainer,
    )

    LOGGER.info("Starting prediction...")
    model: MatBindModule = hydra.utils.instantiate(
        config.model,
        world_size=1,
    )

    predictions = trainer.predict(
        model=model,
        datamodule=datamodule,
        ckpt_path=config.ckpt_path
    )

    LOGGER.info("Aggregating embeddings...")
    aggregated_embeddings = aggregate_embeddings(
        embeddings=predictions,
        modalities=config.data.modalities,
        central_modality=config.data.central_modality,
    )
    LOGGER.info(f"Aggregated embeddings for modality {config.data.central_modality} have shape {aggregated_embeddings[config.data.central_modality][0].shape}")
    for modality in config.data.modalities:
        LOGGER.info(
            f"Aggregated embeddings for modality {modality} have shape {aggregated_embeddings[modality][0].shape}"
        )

    with Path.open(f"{config.embeddings_path}.pkl", "wb") as f:
        pkl.dump(aggregated_embeddings, f, protocol=pkl.HIGHEST_PROTOCOL)
    # with open(f"{config.embeddings_path}.pkl", "rb") as f:
    #     aggregated_embeddings = pkl.load(f)
    #     LOGGER.info(f"loading file from {config.embeddings_path}")
    other_modality = config.data.modalities.copy()
    other_modality.remove(config.data.central_modality)
    retrieval_metrics = full_database_retrieval(
        indices=datamodule.val_data,
        other_modalities=other_modality,
        central_modality=config.data.central_modality,
        embeddings=aggregated_embeddings,
        top_k=config.top_k,
    )
    retrieval_metrics = pd.DataFrame(retrieval_metrics)
    retrieval_metrics.to_csv(f"outputs/retrieval_metrics_{RETRIEVAL_TIME}.csv")
    LOGGER.info(f"Database size: {len(datamodule.val_data)}")
    LOGGER.info(f"Database level retrieval metrics: \n {pformat(retrieval_metrics)}")


@hydra.main(version_base="1.3", config_path="../configs", config_name="retrieval.yaml")
def main(config: DictConfig):
    # torch.backends.cuda.enable_cudnn_sdp(False)
    seed_everything(42)
    embed(config)


if __name__ == "__main__":
    main()
