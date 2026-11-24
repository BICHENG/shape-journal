
import torch
from typing import List, Optional


def softmax(target: torch.Tensor, dim: List[int] = (-1,)):
    dim = [d if d >= 0 else d + target.ndim for d in dim]
    assert sorted(dim) == list(range(min(dim), max(dim) + 1))
    dim_min = min(dim)
    dim_max = max(dim)
    target_flattened = target.flatten(start_dim=dim_min, end_dim=dim_max)
    softmaxed_flattened = torch.softmax(target_flattened, dim=dim_min)
    return softmaxed_flattened.reshape(target.shape)


def soft_argmax(inp: torch.Tensor, dim: List[int] = (-1,)):
    return decode_heatmap(softmax(inp, dim=dim), dim=dim)


def decode_heatmap(inp: torch.Tensor, dim: List[int] = (-1,), output_coord_dim: int = -1):
    result = []
    dim = [d if d >= 0 else d + inp.ndim for d in dim]
    for d in dim:
        # Do NOT replace this with a list comprehension with an if-clause:
        other_heatmap_dims = list(dim)
        other_heatmap_dims.remove(d)
        summed_over_other_heatmap_axes = torch.sum(inp, dim=other_heatmap_dims, keepdim=True)
        coords = linspace(
            0.0, 1.0, inp.shape[d], dtype=inp.dtype, device=summed_over_other_heatmap_axes.device
        )
        decoded = torch.tensordot(summed_over_other_heatmap_axes, coords, dims=([d], [0]))
        x = torch.unsqueeze(decoded, d)
        for hd in sorted(dim)[::-1]:
            x = x.squeeze(hd)
        result.append(x)
    return torch.stack(result, dim=output_coord_dim)


def linspace(
    start: float,
    stop: float,
    num: int,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
    endpoint: bool = True,
):
    start = torch.as_tensor(start, device=device, dtype=dtype)
    stop = torch.as_tensor(stop, device=device, dtype=dtype)

    if endpoint:
        if num == 1:
            return torch.mean(torch.stack([start, stop], dim=0), dim=0, keepdim=True)
        else:
            return torch.linspace(start, stop, num, device=device, dtype=dtype)
    else:
        if num > 1:
            step = (stop - start) / num
            return torch.linspace(start, stop - step, num, device=device, dtype=dtype)
        else:
            return torch.linspace(start, stop, num, device=device, dtype=dtype)
