import datetime
import os

import hydra
import pytorch_lightning as L
import rootutils
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor

# from experiments.train import init_distributed_mode, patch_lightning_slurm_master_addr
from pytorch_lightning.strategies.ddp import DDPStrategy

from matbind.data.components.datasets import DOSDataModule
from matbind.environment.initialization import (
    init_distributed_mode,
    log_distributed_settings,
    patch_lightning_slurm_master_addr,
)
from matbind.models.components.dos_encoder.dos_encoder import DOSLitModule
from matbind.utils.pylogger import get_pylogger

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_DATE = datetime.datetime.now().strftime("%Y%m%d_%H%M")

LOGGER = get_pylogger(__name__)


def train_lit(config: DictConfig):
    run_id = config.run_id + "_" + TRAIN_DATE if hasattr(config, "run_id") else TRAIN_DATE
    # set wandb mode to offline if no WANDB_API_KEY is set
    if not os.getenv("WANDB_API_KEY"):
        os.environ["WANDB_MODE"] = "offline"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        # set PYTORCH_ALLOC_CONF to avoid memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception:
        LOGGER.warning("Your PyTorch version does not support PYTORCH_CUDA_ALLOC_CONF")

    wandb_logger = L.loggers.WandbLogger(save_dir=config.paths.output_dir, project=config.wandb.project, id=run_id)
    world_size = torch.cuda.device_count()

    datamodule: LightningDataModule = DOSDataModule(data_dir=config.data.dataset_path, batch_size=config.data.batch_size)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    model: LightningModule = DOSLitModule(config)
    trainer = L.Trainer(
        max_epochs=config.trainer.max_epochs,
        accelerator=config.trainer.accelerator,
        log_every_n_steps=config.trainer.log_every_n_steps,
        logger=wandb_logger,
        num_nodes=config.trainer.num_nodes,
        devices=world_size if world_size > 1 else "auto",
        strategy=DDPStrategy(find_unused_parameters=True) if world_size > 1 else "auto",
        gradient_clip_val=0.5,
        gradient_clip_algorithm="norm",
        deterministic=True,
        enable_progress_bar=False,
        enable_checkpointing=True,
        precision=config.trainer.precision,
        callbacks=[lr_monitor],
    )
    if config.resume:
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=config.checkpoint_path)
    else:
        trainer.fit(model=model, datamodule=datamodule)

    LOGGER.info("Training complete")
    LOGGER.info("Exiting")


@hydra.main(version_base="1.3", config_path="configs", config_name="dos_lit")
def main(config: DictConfig):
    init_distributed_mode(12354)
    log_distributed_settings(LOGGER)
    train_lit(config)


if __name__ == "__main__":
    patch_lightning_slurm_master_addr()
    seed_everything(42)
    main()
    # train_dos()
