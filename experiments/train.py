import datetime
import os

import hydra
import rootutils
from dotenv import load_dotenv
from lightning.pytorch import Trainer, seed_everything
from omegaconf import DictConfig

from matbind.data.datamodules.matbind import MatBindDataModule
from matbind.environment.initialization import init_distributed
from matbind.model.lightning_module import MatBindModule
from matbind.utils.instantiators import instantiate_callbacks, instantiate_loggers
from matbind.utils.pylogger import get_pylogger
from matbind.utils.rich_utils import print_config_tree

LOGGER = get_pylogger(__name__)

load_dotenv()

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

TRAIN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M")


class Config(DictConfig):
    data: DictConfig
    logger: DictConfig
    callbacks: DictConfig
    trainer: "TrainerConfig"
    model: DictConfig
    ckpt_path: str | None


class TrainerConfig(DictConfig):
    num_nodes: int
    devices: int


def train_matbind(config: Config):
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", (config.trainer.devices * config.trainer.num_nodes)))

    data_module: MatBindDataModule = hydra.utils.instantiate(config.data)

    loggers = instantiate_loggers(config.logger, global_config=config)
    callbacks = instantiate_callbacks(config.callbacks)

    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=loggers,
        callbacks=callbacks,
    )

    model: MatBindModule = hydra.utils.instantiate(
        config.model,
        world_size=WORLD_SIZE,
    )

    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=config.ckpt_path,
    )

    LOGGER.info("Training complete")
    LOGGER.info("Exiting")


@hydra.main(version_base="1.3", config_path="../configs", config_name="matbind.yaml")
def main(config: Config):
    seed_everything(42)
    init_distributed()
    print_config_tree(config)
    train_matbind(config)


if __name__ == "__main__":
    main()
