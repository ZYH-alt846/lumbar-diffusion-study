"""The two final-stage objectives, separated from private experiment I/O.

These are training-step building blocks, not the original scheduling,
checkpoint selection, preprocessing, or an end-to-end reproduction driver.
"""
import torch
from .autoencoder import decode_logits, segmentation_loss
from .diffusion import q_sample, v_target

def diffusion_objective(model, z0, t, noise, buffers, evidence=None):
    zt, eps = q_sample(z0, t, buffers, noise)
    target = v_target(z0, eps, t, buffers)
    prediction = model(zt, t) if evidence is None else model(zt, t, evidence)
    return torch.mean((prediction - target) ** 2)

def deterministic_objective(model, evidence, target, ae, stats):
    from .autoencoder import unstandardize
    code = unstandardize(model(evidence), stats)
    return segmentation_loss(decode_logits(ae, code), target)
