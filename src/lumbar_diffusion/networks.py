"""Shared encoders and compact 3D networks; model bodies match the study."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .geometry import lift_features

FEAT_C = 16
T_32_TO_128 = np.array([[4.,0.,0.,1.5], [0.,4.,0.,1.5],
                        [0.,0.,4.,1.5], [0.,0.,0.,1.]], dtype=np.float32)

def block(dim: int, in_channels: int, out_channels: int) -> nn.Sequential:
    conv = nn.Conv2d if dim == 2 else nn.Conv3d
    groups = 4 if out_channels % 4 == 0 else 1
    return nn.Sequential(
        conv(in_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels), nn.SiLU(),
        conv(out_channels, out_channels, 3, padding=1, bias=False),
        nn.GroupNorm(groups, out_channels), nn.SiLU(),
    )

class ImageEncoder2d(nn.Module):
    """正/侧位 DRR -> (B,2,FEAT_C,64,64)。结构与 baseline.encoder2d 一致。"""
    def __init__(self, c: int = FEAT_C):
        super().__init__()
        self.net = nn.Sequential(block(2, 1, c), nn.AvgPool2d(2),
                                 block(2, c, c), nn.AvgPool2d(2))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        b, v, h, w = image.shape
        f = self.net(image.reshape(b * v, 1, h, w))
        return f.reshape(b, v, -1, *f.shape[-2:])


def condition_volume(enc2d: nn.Module, image: torch.Tensor, matrix128: torch.Tensor):
    """(B,2,256,256) + (B,2,3,4) -> evidence (B,34,32,32,32)。"""
    features = enc2d(image)                                  # (B,2,16,64,64)
    p32 = matrix128 @ torch.from_numpy(T_32_TO_128).to(matrix128.device, matrix128.dtype)
    return lift_features(features, p32, 32, (256, 256))     # (B,34,32,32,32)


class Small3DUNet(nn.Module):
    """小 3D U-Net：encoder-decoder，可选时间嵌入。回归与去噪共用同一结构。"""
    def __init__(self, in_ch: int, out_ch: int, base: int = 32, with_time: bool = False):
        super().__init__()
        self.with_time = with_time
        self.base = base
        self.down0 = block(3, in_ch, base)
        self.down1 = block(3, base, base * 2)
        self.down2 = block(3, base * 2, base * 2)
        self.up1 = block(3, base * 4, base)
        self.up0 = block(3, base * 2, base)
        self.out = nn.Conv3d(base, out_ch, 1)
        if with_time:
            self.time_mlp = nn.Sequential(
                nn.Linear(base, base * 2), nn.SiLU(), nn.Linear(base * 2, base * 2))

    def _time_emb(self, t: torch.Tensor, dim: int) -> torch.Tensor:
        half = dim // 2
        emb = torch.exp(torch.arange(half, device=t.device) *
                        (-np.log(10000.0) / (half - 1)))[None]
        e = t[:, None].float() * emb
        return torch.cat((e.sin(), e.cos()), dim=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None):
        a = self.down0(x)
        b = self.down1(F.avg_pool3d(a, 2))
        c = self.down2(F.avg_pool3d(b, 2))
        if self.with_time:
            te = self.time_mlp(self._time_emb(t, self.base))
            te = te.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            c = c + te
        b_out = self.up1(torch.cat((F.interpolate(c, size=b.shape[2:], mode='trilinear',
                                                   align_corners=False), b), dim=1))
        a_out = self.up0(torch.cat((F.interpolate(b_out, size=a.shape[2:], mode='trilinear',
                                                    align_corners=False), a), dim=1))
        return self.out(a_out)


class DeterministicRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = Small3DUNet(34, 8, base=32, with_time=False)

    def forward(self, evidence):
        return self.unet(evidence)                    # (B,8,32,32,32) 标准化 z
