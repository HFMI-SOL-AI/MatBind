import os
import re

import torch
from lightning.fabric.plugins.environments.slurm import SLURMEnvironment

from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)


def init_distributed(
    master_port: int = 12354,
):
    patch_lightning_slurm_master_addr()
    translate_slurm_env_vars()
    infini_band_master_address(port=master_port)
    configure_pytorch()
    log_distributed_settings()


def get_first_node() -> str:
    """Return the first node we can find in the Slurm node list."""
    nodelist = os.getenv("SLURM_JOB_NODELIST")

    if not nodelist:
        return "localhost"

    bracket_re = re.compile(r"(.*?)\[(.*?)\]")
    dash_re = re.compile("(.*?)-")
    comma_re = re.compile("(.*?),")

    bracket_result = bracket_re.match(nodelist)

    if bracket_result:
        node = bracket_result[1]
        indices = bracket_result[2]

        comma_result = comma_re.match(indices)
        if comma_result:
            indices = comma_result[1]

        dash_result = dash_re.match(indices)
        first_index = dash_result[1] if dash_result else indices

        return node + first_index

    comma_result = comma_re.match(nodelist)
    if comma_result:
        return comma_result[1]

    return nodelist


def translate_slurm_env_vars():
    """Initialize some environment variables for PyTorch Distributed
    using Slurm.
    """
    # The number of total processes started by Slurm.
    try:
        os.environ["WORLD_SIZE"] = os.getenv("SLURM_NTASKS")
        # Index of the current process.
        os.environ["RANK"] = os.getenv("SLURM_PROCID")
        # Index of the current process on this node only.
        os.environ["LOCAL_RANK"] = os.getenv("SLURM_LOCALID")
    except TypeError:
        LOGGER.warning("SLURM environment variables not set.")


def infini_band_master_address(port: int = 12354):
    master_addr = get_first_node()
    if is_running_on_infini_band_cluster():
        master_addr = master_addr + "i"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = master_addr
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = str(port)


def log_distributed_settings():
    LOGGER.info(f"MASTER_ADDR={os.getenv('MASTER_ADDR')}")
    LOGGER.info(f"MASTER_PORT={os.getenv('MASTER_PORT')}")
    LOGGER.info(f"WORLD_SIZE={os.getenv('WORLD_SIZE')}")
    LOGGER.info(f"RANK={os.getenv('RANK')}")
    LOGGER.info(f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')}")


def configure_pytorch():
    torch.set_float32_matmul_precision("medium")

    try:
        # set PYTORCH_ALLOC_CONF to avoid memory fragmentation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    except Exception:
        LOGGER.warning("Your PyTorch version does not support PYTORCH_CUDA_ALLOC_CONF")

    # Set network interface for Gloo backend communication
    # Use InfiniBand interface if on IB cluster, otherwise use default
    if is_running_on_infini_band_cluster():
        # Force Gloo to use InfiniBand interface
        if "GLOO_SOCKET_IFNAME" not in os.environ:
            os.environ["GLOO_SOCKET_IFNAME"] = "ib0"
        if "NCCL_SOCKET_IFNAME" not in os.environ:
            os.environ["NCCL_SOCKET_IFNAME"] = "ib0"


def is_running_on_infini_band_cluster() -> bool:
    """Add more cluster names if needed."""
    return os.getenv("SYSTEMNAME", "") in [
        "juwelsbooster",
        "juwels",
        "jurecadc",
    ]


def patch_lightning_slurm_master_addr():
    if not is_running_on_infini_band_cluster():
        return

    old_resolver = SLURMEnvironment.resolve_root_node_address

    def new_resolver(nodes):
        # Append an i" for communication over InfiniBand.
        return old_resolver(nodes) + "i"

    SLURMEnvironment.__old_resolve_root_node_address = old_resolver
    SLURMEnvironment.resolve_root_node_address = new_resolver
    SLURMEnvironment.resolve_root_node_address = new_resolver
