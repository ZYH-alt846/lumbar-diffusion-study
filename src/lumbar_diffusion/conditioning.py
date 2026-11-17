"""Frozen-prior conditional branch and the matched-topology scratch strategy."""
from __future__ import annotations
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .networks import Small3DUNet, ImageEncoder2d, condition_volume, FEAT_C
BASE = 64
DET_BASE = 64
EVIDENCE_CH = 34
HINT_CH = 8

def time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """与 Small3DUNet._time_emb 一致的正弦时间嵌入（dim=base）。"""
    half = dim // 2
    emb = torch.exp(torch.arange(half, device=t.device) *
                    (-np.log(10000.0) / (half - 1)))[None]
    e = t[:, None].float() * emb
    return torch.cat((e.sin(), e.cos()), dim=1)


class ConditionalPrior(nn.Module):
    """ControlNet 式条件先验去噪器。

    freeze_backbone=True（主方案）：主干冻结，训练 down0'/down1'/down2'/time_mlp' + hint + 三注入 zero-conv。
    freeze_backbone=False（从零对照）：主干与分支都可训练。
    """

    def __init__(self, prior: Small3DUNet, freeze_backbone: bool = True,
                 evidence_ch: int = EVIDENCE_CH, hint_ch: int = HINT_CH):
        super().__init__()
        assert prior.base == BASE and prior.with_time, '必须用 base=64 带时间的先验主干'
        self.base = prior.base
        # 可训练分支：先复制（此时主干仍是 trainable，deepcopy 继承 requires_grad=True）。
        # 必须在冻结之前复制，否则复制品也会带上 requires_grad=False。
        self.down0_t = copy.deepcopy(prior.down0)
        self.down1_t = copy.deepcopy(prior.down1)
        self.down2_t = copy.deepcopy(prior.down2)
        self.time_mlp_t = copy.deepcopy(prior.time_mlp)
        # 锁定主干（共享引用；freeze_backbone 时冻结）
        self.down0 = prior.down0
        self.down1 = prior.down1
        self.down2 = prior.down2
        self.up1 = prior.up1
        self.up0 = prior.up0
        self.out = prior.out
        self.time_mlp = prior.time_mlp
        if freeze_backbone:
            for m in (self.down0, self.down1, self.down2, self.up1, self.up0, self.out, self.time_mlp):
                for p in m.parameters():
                    p.requires_grad_(False)
        # zero-conv（全零初始化）
        self.hint = nn.Conv3d(evidence_ch, hint_ch, 1)        # 34 -> 8
        self.zc_a = nn.Conv3d(self.base, self.base, 1)         # 64 -> 64
        self.zc_b = nn.Conv3d(self.base * 2, self.base * 2, 1)  # 128 -> 128
        self.zc_c = nn.Conv3d(self.base * 2, self.base * 2, 1)  # 128 -> 128
        for m in (self.hint, self.zc_a, self.zc_b, self.zc_c):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, code: torch.Tensor, t: torch.Tensor, evidence: torch.Tensor):
        emb = time_embedding(t, self.base)                     # (B, base)
        # 锁定 encoder（向下路径用原始特征）
        a = self.down0(code)
        b = self.down1(F.avg_pool3d(a, 2))
        c = self.down2(F.avg_pool3d(b, 2))
        # 可训练分支
        h = self.hint(evidence)                                # (B, 8, 32^3)
        a_t = self.down0_t(code + h)
        b_t = self.down1_t(F.avg_pool3d(a_t, 2))
        te_t = self.time_mlp_t(emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        c_t = self.down2_t(F.avg_pool3d(b_t, 2)) + te_t
        # 注入（在加锁定 te 之前）
        a_skip = a + self.zc_a(a_t)
        b_skip = b + self.zc_b(b_t)
        c_bot = c + self.zc_c(c_t)
        te = self.time_mlp(emb).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        c_bot = c_bot + te
        # 锁定 decoder（skip 用注入后的 a_skip/b_skip）
        b_out = self.up1(torch.cat((F.interpolate(c_bot, size=b.shape[2:], mode='trilinear',
                                                   align_corners=False), b_skip), dim=1))
        a_out = self.up0(torch.cat((F.interpolate(b_out, size=a.shape[2:], mode='trilinear',
                                                   align_corners=False), a_skip), dim=1))
        return self.out(a_out)

    def trainable_branch_params(self):
        return (list(self.down0_t.parameters()) + list(self.down1_t.parameters()) +
                list(self.down2_t.parameters()) + list(self.time_mlp_t.parameters()) +
                list(self.hint.parameters()) + list(self.zc_a.parameters()) +
                list(self.zc_b.parameters()) + list(self.zc_c.parameters()))


def build_from_scratch() -> ConditionalPrior:
    """同拓扑从零对照：随机初始化主干 + 复制分支，主干与分支都训练，zero-conv 规则一致。"""
    prior = Small3DUNet(8, 8, base=BASE, with_time=True)
    return ConditionalPrior(prior, freeze_backbone=False)


def build_deterministic() -> Small3DUNet:
    """同条件确定性对照：Small3DUNet(34,8,base=64)，evidence→z→解码（冻结 AE）。"""
    return Small3DUNet(34, 8, base=DET_BASE, with_time=False)


class ConditionalDiffusion(nn.Module):
    """条件生成器整体：ImageEncoder2d（可训练）+ ConditionalPrior（主方案冻结主干）。"""

    def __init__(self, prior: Small3DUNet, freeze_backbone: bool = True):
        super().__init__()
        self.enc2d = ImageEncoder2d(FEAT_C)
        self.denoiser = ConditionalPrior(prior, freeze_backbone=freeze_backbone)

    def evidence(self, image: torch.Tensor, matrix128: torch.Tensor):
        return condition_volume(self.enc2d, image, matrix128)

    def forward(self, code: torch.Tensor, t: torch.Tensor, evidence: torch.Tensor):
        return self.denoiser(code, t, evidence)

class DeterministicConditional(nn.Module):
    """确定性对照：enc2d + Small3DUNet(34,8,base=64,无时间)。"""
    def __init__(self):
        super().__init__()
        self.enc2d = ImageEncoder2d(FEAT_C)
        self.unet = Small3DUNet(34, 8, base=DET_BASE, with_time=False)

    def evidence(self, image, matrix128):
        return condition_volume(self.enc2d, image, matrix128)

    def forward(self, evidence):
        return self.unet(evidence)
