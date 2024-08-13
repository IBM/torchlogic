from typing import Union

import torch
import torch.distributed as dist


def grad_agg(params):
    r"""Aggregate gradients across multiple processes doing training."""
    for p in params:
        if p.requires_grad:
            world_size = dist.get_world_size()
            val = torch.tensor([float(p.grad is not None)])
            votes = float_agg(val)
            assert (votes == world_size) or (votes == 0)
            if p.grad is not None:
                dist.all_reduce(p.grad.data, op=dist.ReduceOp.SUM)


def float_agg(val):
    r"""Aggregate any value val across multiple processes."""
    val = torch.tensor([val])
    dist.all_reduce(val, op=dist.ReduceOp.SUM)
    return val.item()


def tensor_agg(tensor_in: Union[torch.FloatTensor, torch.LongTensor, torch.DoubleTensor, torch.Tensor]):
    r"""Aggregate any tensor by concatenation across multiple processes"""
    world_size = dist.get_world_size()
    tensor_out = [torch.zeros_like(tensor_in.reshape(-1)) for _ in range(world_size)]
    dist.all_gather(tensor_out, tensor_in.reshape(-1))
    tensor_out = torch.cat(tensor_out)
    return tensor_out


__all__ = [grad_agg, float_agg, tensor_agg]