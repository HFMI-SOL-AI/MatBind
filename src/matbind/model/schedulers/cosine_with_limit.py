import math

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from matbind.utils.pylogger import get_pylogger

LOGGER = get_pylogger(__name__)


class CosineWithLimitScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        num_training_steps: int,
        frac_warmup_steps: float = 0.1,
        num_cycles: float = 0.5,
        lower_limit: float = 1e-7,
        last_epoch: int = -1,
    ):
        """
        Initializes the Cosine Annealing Learning Rate Scheduler.

        Args:
            optimizer (Optimizer): The optimizer for which to schedule the learning rate.
            num_warmup_steps (int): The number of steps for the warmup phase.
            num_training_steps (int): The total number of training steps.
            num_cycles (float, optional): The number of waves in the cosine schedule (default is 0.5).
            last_epoch (int, optional): The index of the last epoch when resuming training (default is -1).
        """

        self.num_training_steps = num_training_steps
        self.num_warmup_steps = int(frac_warmup_steps * num_training_steps)
        self.num_cycles = num_cycles
        self.lower_limit = lower_limit
        super().__init__(optimizer, last_epoch)

        LOGGER.info(
            f"Using CosineWithLimitScheduler with {self.num_training_steps} training steps, "
            f"{self.num_warmup_steps} warmup steps, and {num_cycles} cycles."
        )

    def get_lr(self):
        """
        Compute the learning rate based on the current step.
        """
        current_step = self.last_epoch
        if current_step < self.num_warmup_steps:
            return [base_lr * (float(current_step) / float(max(1, self.num_warmup_steps))) for base_lr in self.base_lrs]

        progress = float(current_step - self.num_warmup_steps) / float(max(1, self.num_training_steps - self.num_warmup_steps))
        factor = max(
            self.lower_limit,
            0.5 * (1.0 + math.cos(math.pi * float(self.num_cycles) * 2.0 * progress)),
        )

        return [base_lr * factor for base_lr in self.base_lrs]
