"""Subvoxel label AE, frozen decoding rule, and latent normalization."""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
BIAS = 4.45
K = 5

def voxel_shuffle(x,r=4):
    b,cr,d,h,w=x.shape
    if cr%(r**3):raise ValueError('Channel count must be a multiple of r^3')
    c=cr//r**3
    return x.reshape(b,c,r,r,r,d,h,w).permute(0,1,5,2,6,3,7,4).contiguous().reshape(b,c,d*r,h*r,w*r)

class ResidualBlock(nn.Module):
    def __init__(self, channels=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1),
            nn.GroupNorm(8, channels), nn.SiLU(),
            nn.Conv3d(channels, channels, 3, padding=1))

    def forward(self, x):
        return x + self.net(x)


class SubvoxelAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv3d(6, 32, 4, stride=4)
        self.encoder_block = ResidualBlock()
        self.to_code = nn.Conv3d(32, 8, 1)
        self.from_code = nn.Conv3d(8, 32, 3, padding=1)
        self.decoder_block = ResidualBlock()
        self.head = nn.Sequential(nn.GroupNorm(8, 32), nn.SiLU(), nn.Conv3d(32, 6 * 64, 1))

    def encode(self, labels):
        if labels.ndim != 4 or any(v % 4 for v in labels.shape[-3:]):
            raise ValueError('Expected B,D,H,W with dimensions divisible by four')
        if labels.min() < 0 or labels.max() > 5:
            raise ValueError('Expected labels 0 through 5')
        x = F.one_hot(labels.long(), 6).movedim(-1, 1).to(self.stem.weight.dtype).contiguous()
        return self.to_code(self.encoder_block(self.stem(x))).contiguous()

    def decode(self, code):
        if code.ndim != 5 or code.shape[1] != 8:
            raise ValueError('Expected B,8,D,H,W latent code')
        return voxel_shuffle(self.head(self.decoder_block(self.from_code(code.contiguous()))))

    def forward(self, labels):
        return self.decode(self.encode(labels))

def decode_logits(model: nn.Module, code: torch.Tensor, bias: float = BIAS) -> torch.Tensor:
    """解码 logits，背景通道加固定偏置 bias——损失与最终分类共用同一套 logits。"""
    logits = model.decode(code)
    return torch.cat([logits[:, :1] + float(bias), logits[:, 1:]], dim=1)


def decode_labels(model: nn.Module, code: torch.Tensor, bias: float = BIAS) -> torch.Tensor:
    """code (B,8,32,32,32) -> labels (B,128,128,128)。argmax(带背景偏置 logits)，
    与冻结口径 direct_labels 等价（selfcheck 中已断言）。"""
    return decode_logits(model, code, bias).argmax(1)


def standardize(code: torch.Tensor, stats: dict) -> torch.Tensor:
    mean = torch.tensor(stats['mean'], device=code.device, dtype=code.dtype).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(stats['std'], device=code.device, dtype=code.dtype).reshape(1, -1, 1, 1, 1)
    return (code - mean) / std


def unstandardize(z: torch.Tensor, stats: dict) -> torch.Tensor:
    mean = torch.tensor(stats['mean'], device=z.device, dtype=z.dtype).reshape(1, -1, 1, 1, 1)
    std = torch.tensor(stats['std'], device=z.device, dtype=z.dtype).reshape(1, -1, 1, 1, 1)
    return z * std + mean


def segmentation_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """解码后 CE + 前景类软 Dice（经冻结解码器向编码反向传播）。

    注意：`logits` 必须是 **已加背景偏置的 logits**（`decode_logits`），
    与最终分类 argmax 用同一套 logits，避免训练与推理口径不一致。
    """
    ce = F.cross_entropy(logits, target.long())
    probs = torch.softmax(logits, 1)
    onehot = F.one_hot(target.long(), num_classes=K + 1).permute(0, 4, 1, 2, 3).float()
    num = 2.0 * (probs[:, 1:] * onehot[:, 1:]).sum(dim=(2, 3, 4))
    den = probs[:, 1:].sum(dim=(2, 3, 4)) + onehot[:, 1:].sum(dim=(2, 3, 4)) + 1e-6
    dice = (num / den).mean()
    return ce + (1.0 - dice)
