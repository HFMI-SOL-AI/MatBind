import hydra
from lightning import Callback
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from matbind.utils import pylogger

LOGGER = pylogger.RankedLogger(__name__, rank_zero_only=True)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> list[Callback]:
    """Instantiates callbacks from config.

    :param callbacks_cfg: A DictConfig object containing callback configurations.
    :return: A list of instantiated callbacks.
    """
    callbacks: list[Callback] = []

    if not callbacks_cfg:
        LOGGER.warning("No callback configs found! Skipping..")
        return callbacks

    if not isinstance(callbacks_cfg, DictConfig):
        raise TypeError("Callbacks config must be a DictConfig!")

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            LOGGER.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_loggers(
    logger_cfg: DictConfig,
    global_config: DictConfig | None = None,
) -> list[Logger]:
    """Instantiates loggers from config.

    :param logger_cfg: A DictConfig object containing logger configurations.
    :param global_config: A DictConfig object containing the overall global configuration. This is optionally passed to Logging Services that can log the global configuration on their intialization. For example, WandbLogger.
    Add the corresponding key in LOGGER_TO_CONFIG_KWARG_MAPPING if the logger requires a global configuration.
    :return: A list of instantiated loggers.
    """

    LOGGER_TO_CONFIG_KWARG_MAPPING = {
        "lightning.pytorch.loggers.wandb.WandbLogger": "config",
    }

    if global_config:
        global_config = OmegaConf.to_container(
            global_config,
            resolve=True,
            throw_on_missing=False,
        )  # type: ignore
    loggers: list[Logger] = []

    if not logger_cfg:
        LOGGER.warning("No logger configs found! Skipping...")
        return loggers

    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")

    for _, lg_conf in logger_cfg.items():
        if not isinstance(lg_conf, DictConfig):
            continue

        if "_target_" not in lg_conf:
            continue

        LOGGER.info(f"Instantiating logger <{lg_conf._target_}>")
        logger = hydra.utils.instantiate(lg_conf)

        if "_partial_" in lg_conf:
            logger_name = lg_conf.get("_target_")
            config_kwarg_key = LOGGER_TO_CONFIG_KWARG_MAPPING.get(logger_name)
            kwargs = {config_kwarg_key: global_config} if config_kwarg_key else {}
            try:
                logger = logger(**kwargs)
            except TypeError:
                LOGGER.warning(f"Failed to instantiate logger <{lg_conf._target_}> with global config! Leaving it as is...")
        loggers.append(logger)

    return loggers


def instantiate_none():
    """This function is used to instantiate a None object. It is used by hydra.utils.instantiate to be able to instantiate None."""
    return
