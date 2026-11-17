"""Offline, synthetic interfaces and one gradient step; no patient data."""
from pathlib import Path
import sys
import json
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from lumbar_diffusion.networks import Small3DUNet, ImageEncoder2d, condition_volume
from lumbar_diffusion.conditioning import ConditionalDiffusion, DeterministicConditional
from lumbar_diffusion.autoencoder import SubvoxelAE, decode_labels
from lumbar_diffusion.diffusion import make_noise_buffers, sample_ddpm_cond
from lumbar_diffusion.training import diffusion_objective


def main():
    torch.set_num_threads(2)
    torch.manual_seed(123)
    prior = Small3DUNet(8, 8, base=64, with_time=True)
    model = ConditionalDiffusion(prior, freeze_backbone=True)
    z = torch.randn(1, 8, 32, 32, 32)
    t = torch.tensor([700])
    evidence = torch.randn(1, 34, 32, 32, 32)
    with torch.no_grad():
        equal_at_initialization = torch.equal(model(z, t, evidence), prior(z, t))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    loss = diffusion_objective(model, z, t, torch.randn_like(z), make_noise_buffers(), evidence)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.denoiser.zc_a.weight.grad is not None
    assert all(p.grad is None for p in prior.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    before = model.denoiser.zc_a.weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.denoiser.zc_a.weight)
    # A small synthetic AE volume checks the same stride/shuffle architecture.
    ae = SubvoxelAE().eval()
    with torch.no_grad():
        labels = torch.randint(0, 6, (1, 32, 32, 32))
        code = ae.encode(labels)
        decoded = decode_labels(ae, code)
    assert decoded.shape == labels.shape
    class Zero(torch.nn.Module):
        def forward(self, x, t, evidence):
            return torch.zeros_like(x)
    sample = sample_ddpm_cond(Zero(), torch.zeros(1,34,32,32,32), make_noise_buffers(4),
                             timesteps=4, device=torch.device('cpu'), seed=11, noise_seed=12)
    assert torch.isfinite(sample).all()
    result = {'synthetic_only': True, 'patient_data_read': False,
              'initial_conditional_matches_prior': equal_at_initialization,
              'finite_loss': float(loss.detach()), 'gradient_step_changed_parameters': True,
              'ae_code_shape': list(code.shape), 'sample_shape': list(sample.shape),
              'main_total_parameters': sum(p.numel() for p in model.parameters()),
              'main_trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'deterministic_parameters': sum(p.numel() for p in DeterministicConditional().parameters())}
    assert equal_at_initialization
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
