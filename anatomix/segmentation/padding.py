import numpy as np
import torch


def compute_divisible_spatial_size(spatial_shape, k):
    """
    Compute the new spatial size that is divisible by k.
    """
    new_dim = tuple(int(np.ceil(dim / k) * k) if k > 0 else dim for dim in spatial_shape)
    return new_dim


def compute_pad_width(spatial_size, new_size):
    """Compute the padding width needed to reach new_size from spatial_size."""
    pad_width = []
    for ns_i, sp_i in zip(new_size, spatial_size):
        width = max(ns_i - sp_i, 0)
        pad_width.append((int(width // 2), int(width - (width // 2))))
    return pad_width


def divisible_pad(x, k):
    """Pad x so that its spatial dimensions are divisible by k."""
    new_size = compute_divisible_spatial_size(x.shape, k)
    pad_width = compute_pad_width(x.shape, new_size)
    x2 = np.pad(x, pad_width, mode="constant")
    return x2, pad_width


def divisible_pad_torch(x, k):
    """Pad x so that its spatial dimensions are divisible by k."""
    new_size = compute_divisible_spatial_size(x.shape[2:], k)
    pad_width = compute_pad_width(x.shape[2:], new_size)
    pad_width2 = sum(pad_width, ())
    x2 = torch.nn.functional.pad(x, pad_width2[::-1], mode="constant")
    return x2, pad_width


def unpad_torch(x, pad_width):
    """Remove padding from x according to pad_width."""
    (pz1, pz2), (py1, py2), (px1, px2) = pad_width
    return x[
        ...,
        pz1 if pz1 > 0 else None : -pz2 if pz2 > 0 else None,
        py1 if py1 > 0 else None : -py2 if py2 > 0 else None,
        px1 if px1 > 0 else None : -px2 if px2 > 0 else None,
    ].contiguous()