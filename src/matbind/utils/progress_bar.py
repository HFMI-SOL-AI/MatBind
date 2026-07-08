import logging

from tqdm import tqdm

LOGGER = logging.getLogger(__name__)


class RankZeroOnlyTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        self.rank = self.get_rank()
        kwargs["disable"] = self.rank != 0
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_rank():
        try:
            from lightning.pytorch.utilities import rank_zero_only

            return getattr(rank_zero_only, "rank", 0)
        except ImportError:
            LOGGER.warning(
                "The `rank_zero_only.rank` of `Pytorch Lightning` needs to be set before use.Defaulting to rank 0. If you are in a multi-GPU environment, this results in the progress bar being printed on all ranks."
            )
            return 0
