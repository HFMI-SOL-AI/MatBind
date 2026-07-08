from collections.abc import Callable, Iterator

import torch
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
from torchmetrics.retrieval import RetrievalMRR, RetrievalRecall

from matbind.model.losses import InfoNCELoss, SigLIP
from matbind.model.matbind_model import MatBind
from matbind.utils import clear_cuda_cache
from matbind.utils.pylogger import get_pylogger
from matbind.model.schedulers.cosine_with_limit import CosineWithLimitScheduler

ContrastiveLoss = InfoNCELoss | SigLIP

LOGGER = get_pylogger(__name__)


Forwardable = Tensor | Data
SubBatch = dict[str, Forwardable]
IndexedSubBatch = tuple[SubBatch, int, int]
Batch = dict[str, IndexedSubBatch | SubBatch]
EmbeddingDict = dict[str, Tensor]

Optimizer = Callable[[Iterator[Tensor]], torch.optim.Optimizer]
LRScheduler = (
    Callable[[torch.optim.Optimizer, int | float], torch.optim.lr_scheduler.LRScheduler]
    | Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler]
)


class MatBindModule(LightningModule):
    def __init__(
        self,
        model: MatBind,
        world_size: int,
        loss: ContrastiveLoss,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        batch_size: int,
        learning_rate_multiplier: dict[str, float] | None = None,
        cosine_scheduler_modalities: list[str] | None = None,
        central_modality: str | None = None,
        sync_dist_metrics: bool = True,
        use_single_optimizer: bool = False,
    ):
        super().__init__()

        self.model = model
        self.world_size = world_size
        self.loss = loss
        self.optimizer_ = optimizer
        self.scheduler_ = scheduler

        self.per_device_batch_size = batch_size
        self.batch_size = batch_size * self.world_size
        self.sync_dist_metrics = sync_dist_metrics
        self.use_single_optimizer = use_single_optimizer
        self.automatic_optimization = self.use_single_optimizer
        self.learning_rate_multiplier = learning_rate_multiplier
        self.cosine_scheduler_modalities = cosine_scheduler_modalities
        self.central_modality = central_modality

        LOGGER.info(f"Per device batch size: {self.per_device_batch_size}")
        LOGGER.info(f"Loss batch size: {self.batch_size}")
        LOGGER.info(f"Sync distributed metrics: {self.sync_dist_metrics}")
        LOGGER.info(f"Use single optimizer: {self.use_single_optimizer}")

    def forward(
        self,
        batch: Batch,
    ) -> dict[str, EmbeddingDict]:
        embedding_dicts = {}
        for modality, data in batch.items():
            if data is None:
                continue

            if isinstance(data, tuple):
                # Handle IndexedSubBatch
                data, *_ = data

            LOGGER.debug(f"Forwarding {modality}")

            embedding_dicts[modality] = self.model.forward(data)

        return embedding_dicts

    def _multimodal_loss(
        self,
        embedding_dict: EmbeddingDict,
    ) -> float:
        first_modality, second_modality = embedding_dict.keys()
        z1, z2 = embedding_dict.values()

        LOGGER.debug(self.world_size)
        self._check_nan(z1, "Embedding", first_modality)
        self._check_nan(z2, "Embedding", second_modality)
        LOGGER.debug(f"shape of modality 1 {z1.shape}")
        LOGGER.debug(f"shape of modality 1 {z2.shape}")

        if isinstance(self.loss, SigLIP):
            return self.loss(z1, z2)

        z1 = self.flattened_all_gather(z1)
        z2 = self.flattened_all_gather(z2)
        LOGGER.debug(f"shape of modality 1 {z1.shape}")
        LOGGER.debug(f"shape of modality 1 {z2.shape}")
        return self.loss(z1, z2)

    @staticmethod
    def _check_nan(
        tensor: Tensor,
        tensor_name: str,
        prefix: str,
    ) -> None:
        if torch.isnan(tensor).any():
            LOGGER.error(f"{tensor_name} is nan for {prefix} batch.")

    def _log_loss(self, loss: float, prefix: str) -> None:
        self.log(
            f"{prefix}/loss",
            loss,
            batch_size=self.batch_size,
            sync_dist=self.world_size > 1,
        )

    def _log_temperature(self, modality_keys) -> None:
        modalities = set(modality_keys)
        for modality in list(modalities):
            postprocessor = self.model.postprocessors[modality]
            if not isinstance(postprocessor, torch.nn.Identity) and postprocessor.log_scalor is not None:
                # Check if log_logit_scale is a parameter (learnable) or buffer (non-learnable)
                log_scalor = postprocessor.processor[1]
                is_learnable = isinstance(log_scalor.log_logit_scale, torch.nn.Parameter)
                temp_value = log_scalor.log_logit_scale.exp()

                self.log(
                    f"trainer/temperature_{modality}",
                    temp_value,
                    batch_size=self.batch_size,
                    sync_dist=self.world_size > 1,
                )

                # Debug log to check learnable status
                if self.global_step == 0:
                    LOGGER.info(f"Temperature {modality}: learnable={is_learnable}, value={temp_value.item():.4f}")

    def calculate_grad_norm(self, parameters, norm_type=2):
        """Calculate gradient norm (DDP-safe)"""
        parameters = [p for p in parameters if p.grad is not None]
        if len(parameters) == 0:
            return torch.tensor(0.0, device=self.device)

        device = parameters[0].grad.device
        return torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)

    def on_before_optimizer_step(self, optimizer):
        """Log and clip gradient norm per modality."""
        # calculate gradient norm for each modality
        for modality in self.model.modalities:
            parameter_group = (
                list(self.model.dict_encoders[modality].parameters())
                + list(self.model.dict_projection_heads[modality].parameters())
                + list(self.model.postprocessors[modality].parameters())
            )
            grad_norm_value = self.calculate_grad_norm(parameter_group, norm_type=2)
            self.log(
                f"grad_norm/{modality}",
                grad_norm_value,
                batch_size=self.batch_size,
                sync_dist=self.world_size > 1,
            )

    def _common_step(
        self,
        batch: Batch,
        prefix: str,
    ) -> tuple[dict[str, EmbeddingDict], Tensor]:
        embedding_dicts = self.forward(batch)
        cumulative_loss = 0.0
        for embedding_dict in embedding_dicts.values():
            logging_prefix = f"{prefix}/{'_'.join(embedding_dict.keys())}"
            LOGGER.debug(f"Logging prefix: {logging_prefix}")
            loss = self._multimodal_loss(embedding_dict)
            self._check_nan(loss, "Loss", logging_prefix)
            self._log_loss(loss, logging_prefix)
            cumulative_loss += loss

        cumulative_loss /= len(embedding_dicts)
        self._log_loss(cumulative_loss, prefix)
        self._log_temperature(embedding_dicts.keys())
        return embedding_dicts, cumulative_loss

    @clear_cuda_cache
    def training_step(self, batch: Batch) -> Tensor:
        prefix = "train"
        embedding_dicts, cumulative_loss = self._common_step(batch, prefix)

        if not self.automatic_optimization and isinstance(self.optimizers(), list):
            opts = self.optimizers()
            sch = self.lr_schedulers()
            # Zero all optimizers before backward to ensure clean grads
            for opt in opts:
                opt.zero_grad()

            # Single backward through the full (shared) graph
            self.manual_backward(cumulative_loss)

            # Clip and step per-optimizer (no parameter mutation happens before all backward work is done)
            for opt in opts:
                # Lightning's clip_gradients expects optimizer or params depending on version;
                # here we pass the optimizer object as in original code.
                self.clip_gradients(opt, gradient_clip_val=1.0, gradient_clip_algorithm="norm")
                opt.step()

            # Step schedulers for modalities with CosineWithLimitScheduler
            for modality in self.model.modalities:
                scheduler = sch[self.model.modalities.index(modality)] if isinstance(sch, list) else sch
                if isinstance(scheduler, CosineWithLimitScheduler):
                    scheduler.step()

        return cumulative_loss

    def on_validation_epoch_end(self):
        if not self.automatic_optimization:
            scheds = self.lr_schedulers()
            for modality in self.model.modalities:
                scheduler = scheds[self.model.modalities.index(modality)] if isinstance(scheds, list) else scheds
                if isinstance(scheduler, ReduceLROnPlateau):
                    scheduler.step(self.trainer.callback_metrics[f"valid/{self.central_modality}_{modality}/loss"])

    @clear_cuda_cache
    def validation_step(self, batch: Batch) -> Tensor:
        embedding_dicts, loss = self._common_step(batch, "valid")

        for embedding_dict in embedding_dicts.values():
            first_modality, second_modality = embedding_dict.keys()
            first_embedding, second_embedding = embedding_dict.values()
            self.retrieval_metrics(
                first_embedding,
                second_embedding,
                first_modality,
                second_modality,
                k_list=[1, 5],
                prefix="valid",
            )

        return loss

    def predict_step(
        self,
        batch: IndexedSubBatch,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> EmbeddingDict:
        batch_, *_ = batch
        mid = batch_["crystal_structure"].mid
        return self.model.forward(batch_), mid

    def prediction_aggregate_batches(
        self,
        predictions: list[list[EmbeddingDict]],
    ) -> list[EmbeddingDict]:
        predictions_per_dataloader = []
        for dataloader_predictions in predictions:
            aggregated_embeddings = {k: [] for k in dataloader_predictions[0]}
            for embedding_dict in dataloader_predictions:
                for key, value in embedding_dict.items():
                    aggregated_embeddings[key].append(value)
            predictions_per_dataloader.append({k: torch.cat(v) for k, v in aggregated_embeddings.items()})

        return predictions_per_dataloader

    def configure_optimizers(self):
        base_lr = self.optimizer_.keywords.get("lr", 0.0001)
        optimizer_kwargs = {k: v for k, v in self.optimizer_.keywords.items() if k != "lr" and k != "weight_decay"}

        # Use single optimizer for all modalities if flag is set
        if self.use_single_optimizer:
            # Gather only trainable parameters from all modalities
            all_params = []
            for modality in self.model.modalities:
                modality_params = (
                    list(self.model.dict_encoders[modality].parameters())
                    + list(self.model.dict_projection_heads[modality].parameters())
                    + list(self.model.postprocessors[modality].parameters())
                )
                all_params.extend([p for p in modality_params if p.requires_grad])

            # Create single optimizer for trainable parameters only
            optimizer = torch.optim.AdamW(
                all_params,
                lr=base_lr,
                **optimizer_kwargs,
            )

            # Create single scheduler if specified
            if self.scheduler_ is not None:
                try:
                    scheduler = self.scheduler_(optimizer)
                except TypeError:
                    num_steps = self.trainer.estimated_stepping_batches
                    scheduler = self.scheduler_(optimizer, num_steps)
                    LOGGER.info(f"Scheduler initialized with num_steps={num_steps}")

                return {
                    "optimizer": optimizer,
                    "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                    },
                }

            return {"optimizer": optimizer}

        optimizers = []
        schedulers = []
        # For multi optimizer, the config will specify which optimzier using cosine scheduler and the rest use ReduceLROnPlateau
        # central modality should always using cosine scheduler
        # modality specific weight decay seems does not helo much as well as weighted loss
        # in the config, learning rate multipler can also be specified per modality, by default learing rate for text modality is 0.01, others stay the same
        for modality in self.model.modalities:
            # Gather trainable parameters for this modality only
            modality_params = (
                list(self.model.dict_encoders[modality].parameters())
                + list(self.model.dict_projection_heads[modality].parameters())
                + list(self.model.postprocessors[modality].parameters())
            )
            modality_params = [p for p in modality_params if p.requires_grad]

            # Get modality-specific learning rate
            lr_multiplier = self.learning_rate_multiplier.get(modality, self.learning_rate_multiplier["default"]) if self.learning_rate_multiplier else 1.0
            modality_lr = base_lr * lr_multiplier
            weight_decay = 1e-4

            # Create optimizer for this modality (only trainable params)
            modality_optimizer = torch.optim.AdamW(
                modality_params,
                lr=modality_lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.95),
                **optimizer_kwargs,
            )
            optimizers.append(modality_optimizer)

            # Configure scheduler based on modality
            if self.cosine_scheduler_modalities and modality in self.cosine_scheduler_modalities:
                num_steps = self.trainer.estimated_stepping_batches
                modality_scheduler = CosineWithLimitScheduler(
                    modality_optimizer, num_training_steps=num_steps, frac_warmup_steps=0.1, num_cycles=0.5, lower_limit=1e-7
                )
                schedulers.append(
                    {
                        "scheduler": modality_scheduler,
                        "interval": "step",
                        "frequency": 1,
                    }
                )
            else:
                # Use ReduceLROnPlateau for other modalities
                modality_scheduler = ReduceLROnPlateau(modality_optimizer, mode="min", factor=0.1, patience=3)
                schedulers.append(
                    {
                        "scheduler": modality_scheduler,
                        "interval": "epoch",
                        "frequency": 1,
                        "monitor": f"valid/{self.central_modality}_{modality}/loss",
                    }
                )

        if self.scheduler_ is not None:
            return optimizers, schedulers

        return {"optimizer": optimizers}

    def flattened_all_gather(self, tensor: Tensor) -> Tensor:
        if self.world_size > 1:
            return self.all_gather(tensor, sync_grads=True).flatten(0, 1)
        return tensor

    def retrieval_metrics(
        self,
        embeddings_central_modality: Tensor,
        embeddings_other_modality: Tensor,
        central_modality: str,
        other_modality: str,
        k_list: list[int],
        prefix: str,
    ) -> None:
        with torch.no_grad(), torch.amp.autocast("cuda"):
            # If sync_dist_metrics=False, compute metrics per-device (much more memory efficient slightly less accurate ;))
            if self.world_size > 1 and self.sync_dist_metrics:
                embeddings_central_modality = self.flattened_all_gather(embeddings_central_modality)
                embeddings_other_modality = self.flattened_all_gather(embeddings_other_modality)

            embeddings_central_modality = embeddings_central_modality.contiguous()
            embeddings_other_modality = embeddings_other_modality.contiguous()

            batch_size = embeddings_central_modality.shape[0]
            if batch_size == 0:
                return

            cos_sim = torch.matmul(embeddings_central_modality, embeddings_other_modality.T)

            self._compute_retrieval_metrics_efficient(cos_sim, central_modality, other_modality, k_list, prefix)

    def _compute_retrieval_metrics_efficient(
        self,
        cos_sim: Tensor,
        central_modality: str,
        other_modality: str,
        k_list: list[int],
        prefix: str,
    ) -> None:
        """
        Memory-efficient retrieval metrics computation for large batches.
        Computes metrics directly from the similarity matrix without creating
        the large flattened tensors.

        Uses pessimistic tie-breaking: if there are ties with the correct match,
        counts them as ranking higher (more conservative/accurate metric).
        """
        # Get diagonal similarities (correct matches)
        correct_sims = cos_sim.diagonal()
        num_better = (cos_sim > correct_sims.unsqueeze(1)).sum(dim=1)
        num_ties = (cos_sim == correct_sims.unsqueeze(1)).sum(dim=1) - 1
        ranks = num_better + num_ties + 1

        mrr = (1.0 / ranks.float()).mean()
        self.log(
            f"{prefix}/{central_modality}_{other_modality}/RetrievalMRR",
            mrr,
            batch_size=self.per_device_batch_size * self.world_size,
            sync_dist=True,
        )

        for k in k_list:
            recall_at_k = (ranks <= k).float().mean()
            self.log(
                f"{prefix}/{central_modality}_{other_modality}/RetrievalRecall_top_{k}",
                recall_at_k,
                batch_size=self.per_device_batch_size * self.world_size,
                sync_dist=True,
            )

    def _compute_retrieval_metrics_torchmetrics(
        self,
        cos_sim: Tensor,
        central_modality: str,
        other_modality: str,
        k_list: list[int],
        prefix: str,
    ) -> None:
        """Use torchmetrics for small batches (more memory intensive but validated)."""
        batch_size = cos_sim.shape[0]

        preds = cos_sim.flatten().detach().cpu()
        indexes = torch.arange(batch_size, device=preds.device).repeat_interleave(batch_size)
        target = torch.eye(batch_size, dtype=torch.long, device=preds.device).flatten()

        mrr_metric = RetrievalMRR(sync_on_compute=False).to(preds.device)
        mrr_metric.update(preds, target, indexes=indexes)
        self.log(
            f"{prefix}/{central_modality}_{other_modality}/RetrievalMRR",
            mrr_metric.compute(),
            batch_size=self.per_device_batch_size * self.world_size,
            sync_dist=False,
            rank_zero_only=True,
        )

        for k in k_list:
            recall_metric = RetrievalRecall(top_k=k, sync_on_compute=False).to(preds.device)
            recall_metric.update(preds, target, indexes=indexes)
            self.log(
                f"{prefix}/{central_modality}_{other_modality}/RetrievalRecall_top_{k}",
                recall_metric.compute(),
                batch_size=self.per_device_batch_size * self.world_size,
                sync_dist=False,
                rank_zero_only=True,
            )
