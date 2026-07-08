from functools import wraps
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor


def rename_keys_with_prefix(d: dict, prefix="model.") -> dict:
    new_dict = {}
    for key, value in d.items():
        if key.startswith(prefix):
            # remove the prefix
            new_key = key[len(prefix) :]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict


def select_device() -> str:
    """Selects the device to use for the model."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


def find_all_pairs_in_list(lst: list[Any]) -> list[tuple[Any, Any]]:
    """Finds all pairs in a list."""
    return [(lst[i], lst[j]) for i in range(len(lst)) for j in range(i + 1, len(lst))]


def clear_cuda_cache(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return wrapper


def ring_send_recv(
    tensor: Tensor,
) -> Tensor:
    """Send tensor to next rank and receive tensor from previous rank."""
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    recv_tensor = torch.empty_like(tensor)  # buffer for receiving data from prev

    # Use batch_isend_irecv for proper synchronization
    send_op_list = [dist.P2POp(dist.isend, tensor.contiguous(), next_rank)]
    recv_op_list = [dist.P2POp(dist.irecv, recv_tensor, prev_rank)]

    reqs = dist.batch_isend_irecv(send_op_list + recv_op_list)
    for req in reqs:
        req.wait()

    return recv_tensor


def ring_send_recv_bidir(
    tensor_to_next: Tensor,
    tensor_to_prev: Tensor,
) -> tuple[Tensor, Tensor]:
    """
    Bidirectional ring exchange: send to both neighbors, receive from both.

    Args:
        tensor_to_next: Tensor to send to next rank (rank + 1)
        tensor_to_prev: Tensor to send to previous rank (rank - 1)

    Returns:
        Tuple of (tensor_from_prev, tensor_from_next)
    """
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    next_rank = (rank + 1) % world_size
    prev_rank = (rank - 1 + world_size) % world_size

    recv_from_prev = torch.empty_like(tensor_to_next)
    recv_from_next = torch.empty_like(tensor_to_prev)

    # Send to next, receive from prev
    send_to_next = dist.P2POp(dist.isend, tensor_to_next.contiguous(), next_rank)
    recv_from_prev_op = dist.P2POp(dist.irecv, recv_from_prev, prev_rank)

    # Send to prev, receive from next
    send_to_prev = dist.P2POp(dist.isend, tensor_to_prev.contiguous(), prev_rank)
    recv_from_next_op = dist.P2POp(dist.irecv, recv_from_next, next_rank)

    reqs = dist.batch_isend_irecv([send_to_next, send_to_prev, recv_from_prev_op, recv_from_next_op])
    for req in reqs:
        req.wait()

    return recv_from_prev, recv_from_next
