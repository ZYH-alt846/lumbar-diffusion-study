"""Calibrated projection and feature lifting, with explicit validity masks."""
from __future__ import annotations
import torch
from torch.nn import functional as F

def project_grid(matrices: torch.Tensor, size: int, image_hw: tuple[int, int]):
    """P maps output (x,y,z,1) indices to ORIGINAL INPUT pixel centres.

    Return grid_sample coordinates for align_corners=False, plus validity.
    Keep projection math in float32 even when convolutions use mixed precision.
    """
    if matrices.ndim != 4 or matrices.shape[1:] != (2, 3, 4):
        raise ValueError("P_grid_px must have shape [B,2,3,4]")
    height, width = image_hw
    with torch.autocast(device_type=matrices.device.type, enabled=False):
        axis = torch.arange(size, device=matrices.device, dtype=torch.float32)
        z, y, x = torch.meshgrid(axis, axis, axis, indexing="ij")
        points = torch.stack((x, y, z, torch.ones_like(x)), dim=-1).reshape(-1, 4)
        projected = torch.einsum("bvij,nj->bvni", matrices.float(), points)
        depth = projected[..., 2]
        uv = projected[..., :2] / depth.clamp_min(1e-6).unsqueeze(-1)
        u, v = uv.unbind(-1)
        valid = ((depth > 1e-6) & torch.isfinite(uv).all(-1) &
                 (u >= 0) & (u <= width - 1) & (v >= 0) & (v <= height - 1))
        grid = torch.stack((2 * (u + .5) / width - 1,
                            2 * (v + .5) / height - 1), dim=-1)
        grid = torch.where(valid.unsqueeze(-1), grid, torch.full_like(grid, 2))
    return grid, valid


def lift_features(features: torch.Tensor, matrices: torch.Tensor,
                  size: int, image_hw: tuple[int, int]) -> torch.Tensor:
    """Lift [B,2,C,Hf,Wf] features into two aligned 3D feature volumes.

    Encoder downsampling uses average pooling (pixel-centre consistent), so
    normalized coordinates refer to the same physical ray at each resolution.
    Out-of-frame/behind-camera samples are zeroed and have explicit masks.
    """
    batch, views, channels, height, width = features.shape
    grid, valid = project_grid(matrices, size, image_hw)
    with torch.autocast(device_type=features.device.type, enabled=False):
        sampled = F.grid_sample(
            features.float().reshape(batch * views, channels, height, width),
            grid.reshape(batch * views, size * size, size, 2),
            mode="bilinear", padding_mode="border", align_corners=False,
        ).reshape(batch, views, channels, size, size, size)
        masks = valid.reshape(batch, views, 1, size, size, size).float()
        sampled = sampled * masks
    return torch.cat((sampled.flatten(1, 2), masks.flatten(1, 2)), dim=1)
