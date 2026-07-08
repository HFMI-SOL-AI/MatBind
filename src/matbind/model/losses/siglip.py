import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from matbind.utils.utils import ring_send_recv, ring_send_recv_bidir


class SigLIP(nn.Module):
    """
    Sigmoid loss for Language-Image Pre-training(SigLIP)
    Ref: http://arxiv.org/abs/2303.15343

    Info: The bidirectional ring exchange implementation is copied and modified from open_cip
    https://github.com/mlfoundations/open_clip/blob/fc5a37b72d705f760ebbc7915b84729816ed471f/src/open_clip/loss.py#L239

    Args:
        temperature: Temperature parameter for scaling logits
        bias: Bias term added to logits
        symmetric: Whether to compute loss in both directions and average
        sparse: Whether to add L1 regularization
        l1_norm_weight: Weight for L1 regularization term
        reduction: How to reduce loss ("mean" or "sum")
        chunk_size: If specified and < batch_size, enables chunked computation with
                   gradient checkpointing. This reduces peak memory at the cost of
                   some additional compute. Setting to None disables chunking.
                   Hypothetical example: batch_size=128, chunk_size=64 → ~50% memory reduction
        bidir: Whether to use bidirectional ring communication. When True, reduces
               communication rounds from world_size to ⌈(world_size-1)/2⌉, improving
               efficiency on multi-node setups. Default: True

    """

    def __init__(
        self,
        temperature: float = 1.00,
        bias: float = -10.0,
        symmetric: bool = True,
        sparse: bool = False,
        l1_norm_weight: float = 0.0,
        reduction: str = "mean",
        chunk_size: int | None = None,
        bidir: bool = True,
    ):
        super().__init__()
        self.temperature = temperature
        self.bias = nn.Parameter(torch.tensor(bias))
        self.symmetric = symmetric
        self.sparse = sparse
        self.l1_norm_weight = l1_norm_weight
        self.reduction = reduction
        self.chunk_size = chunk_size
        self.bidir = bidir

    def forward(
        self,
        z1_local: Tensor,
        z2_local: Tensor,
    ) -> Tensor:
        """
        Compute SigLIP loss using ring rotation with optional chunked computation.

        Args:
            z1_local: Local embeddings for modality 1, shape [batch_size, embed_dim]
            z2_local: Local embeddings for modality 2, shape [batch_size, embed_dim]

        Returns:
            Scalar loss value
        """
        if z1_local.ndim != 2 or z2_local.ndim != 2:
            raise ValueError("Input tensors must be 2-dimensional")
        if z1_local.shape[0] != z2_local.shape[0]:
            raise ValueError("Input tensors must have the same batch size")
        if z1_local.shape[1] != z2_local.shape[1]:
            raise ValueError("Input tensors must have the same feature dimension")

        loss = self._compute_ring_loss(z1_local, z2_local)

        if self.symmetric:
            if dist.is_initialized():
                dist.barrier()
            loss_reverse = self._compute_ring_loss(z2_local, z1_local)
            loss = (loss + loss_reverse) / 2

        if self.sparse:
            loss = loss + self.l1_norm_weight * torch.norm(z1_local)

        return loss

    def _compute_ring_loss(self, z1_local: Tensor, z2_local: Tensor) -> Tensor:
        """
        Unified ring-based computation (handles both chunked and non-chunked).
        z1_local stays on the device but z2 rotates in the ring.
        """
        if not dist.is_initialized():
            return self._compute_local_loss(z1_local, z2_local)

        world_size, rank = dist.get_world_size(), dist.get_rank()
        dist.barrier()

        if self.bidir:
            total_loss, num_elements = self._run_bidir_ring(z1_local, z2_local, world_size, rank)
        else:
            total_loss, num_elements = self._run_unidir_ring(z1_local, z2_local, world_size, rank)

        return self._finalize_loss(total_loss, num_elements)

    def _compute_local_loss(self, z1: Tensor, z2: Tensor) -> Tensor:
        """Compute loss for non-distributed (local) case."""
        loss_value, num_elements = self._compute_step_loss(z1, z2, z1_start_idx=0, z2_start_idx=0)

        if self.reduction == "mean":
            return loss_value / num_elements
        elif self.reduction == "sum":
            return loss_value
        else:
            return loss_value / num_elements

    def _run_bidir_ring(self, z1_local: Tensor, z2_local: Tensor, world_size: int, rank: int) -> tuple[Tensor, int]:
        """Execute bidirectional ring exchange (reduces communication by ~50%)."""
        local_batch_size = z1_local.shape[0]
        z1_global_start = rank * local_batch_size
        total_loss = torch.tensor(0.0, device=z1_local.device, dtype=z1_local.dtype)
        num_elements = 0

        num_bidir_steps, remainder = divmod(world_size - 1, 2)

        # Step 0: compute with local z2
        step_loss, step_elems = self._compute_step_loss(
            z1_local, z2_local, z1_start_idx=z1_global_start, z2_start_idx=z1_global_start
        )
        total_loss, num_elements = self._accumulate_loss(total_loss, num_elements, step_loss, step_elems)

        z2_to_prev = z2_to_next = z2_local

        # Bidirectional exchanges
        for step in range(1, num_bidir_steps + 1):
            z2_from_prev, z2_from_next = ring_send_recv_bidir(z2_to_next, z2_to_prev)

            # Compute loss for both received tensors
            z2_prev_rank = (rank - step) % world_size
            z2_prev_start = z2_prev_rank * local_batch_size
            loss_prev, elem_prev = self._compute_step_loss(
                z1_local, z2_from_prev, z1_start_idx=z1_global_start, z2_start_idx=z2_prev_start
            )

            z2_next_rank = (rank + step) % world_size
            z2_next_start = z2_next_rank * local_batch_size
            loss_next, elem_next = self._compute_step_loss(
                z1_local, z2_from_next, z1_start_idx=z1_global_start, z2_start_idx=z2_next_start
            )

            total_loss, num_elements = self._accumulate_loss(
                total_loss, num_elements, loss_prev + loss_next, elem_prev + elem_next
            )

            # Update for next iteration - swap to continue counter-rotating rings
            z2_to_prev = z2_from_next
            z2_to_next = z2_from_prev

        # Handle remainder (if world_size is even)
        if remainder:
            z2_from_prev = ring_send_recv(z2_to_next)
            z2_source_rank = (rank - (num_bidir_steps + 1)) % world_size
            z2_source_start = z2_source_rank * local_batch_size

            step_loss, step_elems = self._compute_step_loss(
                z1_local, z2_from_prev, z1_start_idx=z1_global_start, z2_start_idx=z2_source_start
            )
            total_loss, num_elements = self._accumulate_loss(total_loss, num_elements, step_loss, step_elems)

        return total_loss, num_elements

    def _run_unidir_ring(self, z1_local: Tensor, z2_local: Tensor, world_size: int, rank: int) -> tuple[Tensor, int]:
        """Execute unidirectional ring exchange."""
        local_batch_size = z1_local.shape[0]
        z1_global_start = rank * local_batch_size
        total_loss = torch.tensor(0.0, device=z1_local.device, dtype=z1_local.dtype)
        num_elements = 0

        z2_current = z2_local.clone()

        for step in range(world_size):
            z2_source_rank = (rank - step) % world_size
            z2_global_start = z2_source_rank * local_batch_size

            step_loss, step_elems = self._compute_step_loss(
                z1_local, z2_current, z1_start_idx=z1_global_start, z2_start_idx=z2_global_start
            )
            total_loss, num_elements = self._accumulate_loss(total_loss, num_elements, step_loss, step_elems)

            # Ring send/receive for next iteration (except last)
            if step < world_size - 1:
                z2_current = ring_send_recv(z2_current)

        return total_loss, num_elements

    def _compute_step_loss(self, z1: Tensor, z2: Tensor, z1_start_idx: int, z2_start_idx: int) -> tuple[Tensor, int]:
        """
        Compute loss for a single ring step (dispatches to chunked or non-chunked).
        Returns (loss_sum, num_elements) for proper averaging.
        """
        batch_size = z1.shape[0]
        use_chunking = self.chunk_size is not None and self.chunk_size < batch_size

        if use_chunking:
            return self._compute_loss_chunked(z1, z2, z1_start_idx, z2_start_idx)
        else:
            block_loss = self._compute_block_loss(z1, z2, z1_start_idx, z2_start_idx)
            return block_loss.sum(), block_loss.numel()

    def _accumulate_loss(self, total_loss: Tensor, num_elements: int, step_loss: Tensor, step_elems: int) -> tuple[Tensor, int]:
        """Accumulate loss based on reduction mode."""
        if self.reduction == "mean":
            return total_loss + step_loss, num_elements + step_elems
        elif self.reduction == "sum":
            return total_loss + step_loss, num_elements
        else:
            return total_loss + step_loss, num_elements + step_elems

    def _finalize_loss(self, total_loss: Tensor, num_elements: int) -> Tensor:
        """Finalize loss: reduce locally, then all-reduce across ranks."""
        local_loss = total_loss / num_elements if self.reduction == "mean" else total_loss

        dist.all_reduce(local_loss, op=dist.ReduceOp.AVG)
        return local_loss

    def _compute_block_loss(
        self,
        z1: Tensor,
        z2: Tensor,
        z1_start_idx: int,
        z2_start_idx: int,
    ) -> Tensor:
        """Compute loss for a block."""
        batch1, batch2 = z1.shape[0], z2.shape[0]

        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        logits = z1 @ z2.T / self.temperature + self.bias
        labels = torch.full((batch1, batch2), -1.0, device=z1.device, dtype=logits.dtype)

        # Mark positive pairs (diagonal elements in global space)
        overlap_start = max(z1_start_idx, z2_start_idx)
        overlap_end = min(z1_start_idx + batch1, z2_start_idx + batch2)

        if overlap_start < overlap_end:
            for global_idx in range(overlap_start, overlap_end):
                local_i = global_idx - z1_start_idx
                local_j = global_idx - z2_start_idx
                labels[local_i, local_j] = 1.0

        return -F.logsigmoid(labels * logits)

    def _compute_loss_chunked(
        self,
        z1: Tensor,
        z2: Tensor,
        z1_start_idx: int,
        z2_start_idx: int,
    ) -> tuple[Tensor, int]:
        """
        Compute loss with chunked matrix multiplication and gradient checkpointing.

        Chunks both z1 (rows) and z2 (columns) to reduce peak memory during
        similarity matrix computation. Uses gradient checkpointing to trade
        compute for memory.

        Args:
            z1: First embedding tensor, shape [batch1, dim]
            z2: Second embedding tensor, shape [batch2, dim]
            z1_start_idx: Global start index for z1 (for positive pair alignment)
            z2_start_idx: Global start index for z2 (for positive pair alignment)

        Returns:
            Tuple of (loss_sum, num_elements) for proper averaging
        """

        batch1, batch2 = z1.shape[0], z2.shape[0]
        chunk_size = self.chunk_size if self.chunk_size is not None else batch1

        # Normalize once (outside checkpoint to avoid redundant computation)
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)

        total_loss = torch.tensor(0.0, device=z1.device, dtype=z1.dtype)
        total_elements = 0

        # Chunk z1 (rows)
        for i in range(0, batch1, chunk_size):
            i_end = min(i + chunk_size, batch1)
            z1_chunk = z1[i:i_end]

            # Chunk z2 (columns)
            for j in range(0, batch2, chunk_size):
                j_end = min(j + chunk_size, batch2)
                z2_chunk = z2[j:j_end]

                # Use gradient checkpointing for this chunk
                # This recomputes logits during backward instead of storing them
                chunk_loss = checkpoint(
                    self._compute_chunk_loss,
                    z1_chunk,
                    z2_chunk,
                    torch.tensor(z1_start_idx + i),
                    torch.tensor(z2_start_idx + j),
                    torch.tensor(i_end - i),
                    torch.tensor(j_end - j),
                    use_reentrant=False,
                )

                total_loss += chunk_loss.sum()
                total_elements += chunk_loss.numel()

        return total_loss, total_elements

    def _compute_chunk_loss(
        self,
        z1_chunk: Tensor,
        z2_chunk: Tensor,
        chunk_z1_start: Tensor,
        chunk_z2_start: Tensor,
        chunk_i_size: Tensor,
        chunk_j_size: Tensor,
    ) -> Tensor:
        """
        Compute loss for a single chunk (called by checkpoint).

        This function is checkpointed, meaning its intermediate tensors
        (logits, labels) are not stored during forward pass and are
        recomputed during backward pass.

        Args:
            z1_chunk: Normalized z1 chunk, shape [chunk_i_size, dim]
            z2_chunk: Normalized z2 chunk, shape [chunk_j_size, dim]
            chunk_z1_start: Global start index for this z1 chunk
            chunk_z2_start: Global start index for this z2 chunk
            chunk_i_size: Size of z1 chunk
            chunk_j_size: Size of z2 chunk

        Returns:
            Loss tensor for this chunk, shape [chunk_i_size, chunk_j_size]
        """
        # Convert tensor indices back to int (checkpoint requires tensor args)
        chunk_z1_start = int(chunk_z1_start.item())
        chunk_z2_start = int(chunk_z2_start.item())
        chunk_i_size = int(chunk_i_size.item())
        chunk_j_size = int(chunk_j_size.item())

        # Compute logits for this chunk
        logits = (z1_chunk @ z2_chunk.T) / self.temperature + self.bias

        # Create labels (all negatives by default)
        labels = torch.full((chunk_i_size, chunk_j_size), -1.0, device=z1_chunk.device, dtype=logits.dtype)

        # Mark positive pairs (diagonal elements in global space)
        overlap_start = max(chunk_z1_start, chunk_z2_start)
        overlap_end = min(chunk_z1_start + chunk_i_size, chunk_z2_start + chunk_j_size)

        if overlap_start < overlap_end:
            for global_idx in range(overlap_start, overlap_end):
                local_i = global_idx - chunk_z1_start
                local_j = global_idx - chunk_z2_start
                labels[local_i, local_j] = 1.0

        # Return element-wise loss (will be summed outside)
        return -F.logsigmoid(labels * logits)
