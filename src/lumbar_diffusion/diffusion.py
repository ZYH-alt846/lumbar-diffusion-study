"""v-prediction and the study's DDIM/DDPM paths; zero-indexed time."""
from __future__ import annotations
import numpy as np
import torch

def make_noise_buffers(timesteps: int = 1000) -> dict:
    betas = torch.linspace(1e-4, 0.02, timesteps)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, 0)
    return {
        'betas': betas,
        'sqrt_alpha_bar': torch.sqrt(alpha_bar),
        'sqrt_one_minus_alpha_bar': torch.sqrt(1.0 - alpha_bar),
    }


def q_sample(z0, t, bufs, noise=None):
    """前向加噪 z_t = √ᾱ_t·z0 + √(1−ᾱ_t)·ε，返回 (z_t, ε)。"""
    noise = torch.randn_like(z0) if noise is None else noise
    a = bufs['sqrt_alpha_bar'][t].reshape(-1, 1, 1, 1, 1)
    b = bufs['sqrt_one_minus_alpha_bar'][t].reshape(-1, 1, 1, 1, 1)
    return a * z0 + b * noise, noise


def v_target(z0, eps, t, bufs):
    """v = α·ε − σ·z0（α=√ᾱ, σ=√(1−ᾱ)）。"""
    a = bufs['sqrt_alpha_bar'][t].reshape(-1, 1, 1, 1, 1)
    b = bufs['sqrt_one_minus_alpha_bar'][t].reshape(-1, 1, 1, 1, 1)
    return a * eps - b * z0


def ddim_sequence(timesteps: int = 1000, steps: int = 100) -> np.ndarray:
    """递减去噪时间点（含最后一跳 t->-1 干净终点）。"""
    return np.unique(np.linspace(0, timesteps - 1, steps).astype(np.int64))[::-1]


@torch.no_grad()
def sample_ddim_uncond(model, bufs, steps: int = 100, device=None, seed=None) -> torch.Tensor:
    """采样单个样本：N(0,I)（Generator(seed)）-> DDIM eta=0 -> 干净 z0。

    返回 (1,8,32,32,32) 标准化码；解码由调用方 unstandardize + decode(AE, bias) + argmax。
    """
    device = device if device is not None else next(model.parameters()).device
    seq = ddim_sequence(1000, steps)
    alpha_bar = bufs['sqrt_alpha_bar'] ** 2
    one = torch.tensor(1.0, device=device)

    def ab(t: int):
        return one if t < 0 else alpha_bar[t]

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    code = torch.randn(1, 8, 32, 32, 32, device=device, generator=gen)
    for k in range(len(seq)):
        t = int(seq[k])
        t_prev = int(seq[k + 1]) if k + 1 < len(seq) else -1
        tt = torch.full((1,), t, device=device, dtype=torch.long)
        out = model(code, tt)
        a = bufs['sqrt_alpha_bar'][tt].reshape(-1, 1, 1, 1, 1)
        b = bufs['sqrt_one_minus_alpha_bar'][tt].reshape(-1, 1, 1, 1, 1)
        x0 = a * code - b * out
        eps = b * code + a * out
        if t_prev < 0:
            code = x0                        # 干净终点：ᾱ=1，输出预测 x0
        else:
            abp = ab(t_prev)
            code = torch.sqrt(abp) * x0 + torch.sqrt(1.0 - abp) * eps
    return code

def ddpm_posterior(bufs: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回 (coef1, coef2, sigma)，各 (T,)（在 bufs 所在 device）。

    coef1 = √ᾱ_{i-1}·β_i/(1−ᾱ_i)   （乘 x0）
    coef2 = √α_i·(1−ᾱ_{i-1})/(1−ᾱ_i)（乘 z_i）
    sigma = √(β_i·(1−ᾱ_{i-1})/(1−ᾱ_i))（后验标准差 β̃_i）
    0-indexed；ᾱ_{-1}=1（i=0 时 coef1=1、coef2=0、sigma=0 → 干净终点）。
    """
    betas = bufs['betas']
    alphas = 1.0 - betas
    alpha_bar = bufs['sqrt_alpha_bar'] ** 2
    ab_prev = torch.cat([torch.ones(1, device=betas.device, dtype=betas.dtype),
                         alpha_bar[:-1]], dim=0)
    one_m_ab = 1.0 - alpha_bar
    coef1 = torch.sqrt(ab_prev) * betas / one_m_ab
    coef2 = torch.sqrt(alphas) * (1.0 - ab_prev) / one_m_ab
    sigma = torch.sqrt(betas * (1.0 - ab_prev) / one_m_ab)
    return coef1, coef2, sigma


@torch.no_grad()
def sample_ddpm_uncond(model, bufs: dict, timesteps: int = 1000, device=None,
                       seed: int | None = None, noise_seed: int | None = None) -> torch.Tensor:
    """随机反向（DDPM）采样单个样本，返回 (1,8,32,32,32) 标准化码。

    - 初始噪声 x_T 用 Generator(seed)（与 sample_ddim_uncond 相同构造 → 同 seed 同 x_T）。
    - 每步额外噪声 ε 用 Generator(noise_seed) 顺序抽取（固定并记录种子，保证可复现）。
    - 干净终点 i=0 输出 x0，不加噪。
    """
    device = device if device is not None else next(model.parameters()).device
    assert bufs['betas'].numel() == timesteps, (bufs['betas'].numel(), timesteps)
    coef1, coef2, sigma = ddpm_posterior(bufs)
    sqrt_ab = bufs['sqrt_alpha_bar']
    sqrt_omab = bufs['sqrt_one_minus_alpha_bar']

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    code = torch.randn(1, 8, 32, 32, 32, device=device, generator=gen)  # x_T ~ N(0,I)

    ngen = None
    if noise_seed is not None:
        ngen = torch.Generator(device=device)
        ngen.manual_seed(noise_seed)

    for i in range(timesteps - 1, -1, -1):
        tt = torch.full((1,), i, device=device, dtype=torch.long)
        v = model(code, tt)
        a = sqrt_ab[i].reshape(1, 1, 1, 1, 1)
        b = sqrt_omab[i].reshape(1, 1, 1, 1, 1)
        x0 = a * code - b * v
        mu = coef1[i] * x0 + coef2[i] * code
        if i > 0:
            eps = torch.randn(1, 8, 32, 32, 32, device=device, generator=ngen)
            code = mu + sigma[i] * eps
        else:
            code = x0   # 干净终点直接返回预测 x0（float32 下 coef1[0]≈0.999834，非精确 1，不能乘它）
    return code

@torch.no_grad()
def sample_ddpm_cond(model, evidence: torch.Tensor, bufs: dict, timesteps: int = 1000,
                     device=None, seed: int | None = None,
                     noise_seed: int | None = None) -> torch.Tensor:
    """条件 DDPM 随机反向采样单个样本，返回 (1,8,32,32,32) 标准化码。

    - evidence (1,34,32,32,32) 预先算好，循环内只调用 denoiser(model)。
    - 初始噪声 x_T 用 Generator(seed)；每步额外噪声用 Generator(noise_seed) 顺序抽取。
    - 干净终点 i=0 直接返回 x0（不乘 float32 coef1[0]≈0.999834）。
    """
    device = device if device is not None else next(model.parameters()).device
    assert bufs['betas'].numel() == timesteps, (bufs['betas'].numel(), timesteps)
    assert evidence.ndim == 5 and evidence.shape[1] == 34, tuple(evidence.shape)
    B = evidence.shape[0]
    coef1, coef2, sigma = ddpm_posterior(bufs)
    sqrt_ab = bufs['sqrt_alpha_bar']
    sqrt_omab = bufs['sqrt_one_minus_alpha_bar']

    gen = None
    if seed is not None:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
    code = torch.randn(B, 8, 32, 32, 32, device=device, generator=gen)

    ngen = None
    if noise_seed is not None:
        ngen = torch.Generator(device=device)
        ngen.manual_seed(noise_seed)

    for i in range(timesteps - 1, -1, -1):
        tt = torch.full((B,), i, device=device, dtype=torch.long)
        v = model(code, tt, evidence)
        a = sqrt_ab[i].reshape(1, 1, 1, 1, 1)
        b = sqrt_omab[i].reshape(1, 1, 1, 1, 1)
        x0 = a * code - b * v
        mu = coef1[i] * x0 + coef2[i] * code
        if i > 0:
            eps = torch.randn(B, 8, 32, 32, 32, device=device, generator=ngen)
            code = mu + sigma[i] * eps
        else:
            code = x0
    return code
