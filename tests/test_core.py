"""Synthetic correctness checks, not experiment-performance tests."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from lumbar_diffusion.geometry import project_grid, lift_features
from lumbar_diffusion.networks import Small3DUNet, condition_volume, ImageEncoder2d
from lumbar_diffusion.conditioning import ConditionalDiffusion, DeterministicConditional
from lumbar_diffusion.autoencoder import SubvoxelAE, decode_logits, decode_labels, standardize, unstandardize
from lumbar_diffusion.diffusion import make_noise_buffers, q_sample, v_target, sample_ddpm_cond, sample_ddpm_uncond
from lumbar_diffusion.bridge import BrownianBridge
from lumbar_diffusion.metrics import evaluate_labels

torch.set_num_threads(2)

class CoreTests(unittest.TestCase):
    def test_v_roundtrip(self):
        torch.manual_seed(10)
        x0 = torch.randn(2, 8, 3, 3, 3)
        eps = torch.randn_like(x0)
        t = torch.tensor([0, 999])
        bufs = make_noise_buffers()
        xt, _ = q_sample(x0, t, bufs, eps)
        v = v_target(x0, eps, t, bufs)
        a = bufs['sqrt_alpha_bar'][t].reshape(2,1,1,1,1)
        b = bufs['sqrt_one_minus_alpha_bar'][t].reshape(2,1,1,1,1)
        self.assertTrue(torch.allclose(a*xt-b*v, x0, atol=1e-6))
        self.assertTrue(torch.allclose(b*xt+a*v, eps, atol=1e-6))

    def test_conditional_and_unconditional_sampler(self):
        class Uncond(torch.nn.Module):
            def forward(self, x, t): return torch.zeros_like(x)
        class Cond(torch.nn.Module):
            def forward(self, x, t, evidence): return torch.zeros_like(x)
        bufs=make_noise_buffers(4)
        args=dict(timesteps=4,device=torch.device('cpu'),seed=9,noise_seed=10)
        a=sample_ddpm_uncond(Uncond(),bufs,**args)
        b=sample_ddpm_cond(Cond(),torch.zeros(1,34,32,32,32),bufs,**args)
        c=sample_ddpm_cond(Cond(),torch.zeros(1,34,32,32,32),bufs,**args)
        self.assertTrue(torch.equal(a,b) and torch.equal(b,c))

    def test_bridge_oracle_all_jumps(self):
        bb=BrownianBridge(1000)
        x0=torch.randn(1,8,3,3,3,dtype=torch.float64)
        y=torch.randn_like(x0)
        for steps in [1,5,10]:
            out=bb.sample(y,lambda xt,t: xt-x0,steps=steps,
                          generator=torch.Generator().manual_seed(123))
            self.assertLess(float((out-x0).abs().max()),1e-14)

    def test_projection_order_and_pixel_centers(self):
        # x projects to u, y to v, homogeneous depth remains positive.
        mat=torch.tensor([[[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]],
                            [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]]]])
        grid,valid=project_grid(mat,2,(4,4))
        self.assertTrue(valid.all())
        self.assertTrue(torch.allclose(grid[0,0,0],torch.tensor([-.75,-.75])))
        self.assertTrue(torch.allclose(grid[0,0,1],torch.tensor([-.25,-.75])))

    def test_lift_validity_masks(self):
        mat=torch.tensor([[[[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]],
                            [[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,0.,1.]]]])
        lifted=lift_features(torch.ones(1,2,3,4,4),mat,2,(4,4))
        self.assertEqual(list(lifted.shape),[1,8,2,2,2])
        self.assertTrue(torch.equal(lifted,torch.ones_like(lifted)))

    def test_zero_injection_and_freezing(self):
        prior=Small3DUNet(8,8,base=64,with_time=True)
        model=ConditionalDiffusion(prior,True)
        z=torch.randn(1,8,8,8,8);t=torch.tensor([500]);e=torch.randn(1,34,8,8,8)
        self.assertTrue(torch.equal(prior(z,t),model(z,t,e)))
        model(z,t,e).square().mean().backward()
        self.assertTrue(all(p.grad is None for p in prior.parameters()))
        self.assertIsNotNone(model.denoiser.zc_a.weight.grad)
        self.assertTrue(all(p.requires_grad for p in model.denoiser.down0_t.parameters()))

    def test_stats_and_decode_decision(self):
        ae=SubvoxelAE()
        labels=torch.randint(0,6,(1,16,16,16))
        z=ae.encode(labels)
        stats={'mean':[0.1]*8,'std':[0.5]*8}
        self.assertTrue(torch.allclose(unstandardize(standardize(z,stats),stats),z,atol=1e-7))
        self.assertTrue(torch.equal(decode_logits(ae,z).argmax(1),decode_labels(ae,z)))

    def test_missing_prediction_is_not_dropped(self):
        gt=np.zeros((12,12,12),dtype=np.uint8)
        for k in range(1,6):gt[k*2:k*2+1,2:4,2:4]=k
        pred=gt.copy();pred[pred==3]=0
        out=evaluate_labels(pred,gt)
        self.assertEqual(out['miss_count'],1)
        self.assertEqual(out['per_vertebra'][2]['dice'],0)
        self.assertIsNone(out['per_vertebra'][2]['hd95_mm'])
        self.assertAlmostEqual(out['dice'],0.8)

    def test_model_capacity_is_registered(self):
        main=ConditionalDiffusion(Small3DUNet(8,8,base=64,with_time=True),True)
        self.assertEqual(sum(p.numel() for p in main.parameters()),4328048)
        self.assertEqual(sum(p.numel() for p in main.parameters() if p.requires_grad),1743464)
        self.assertEqual(sum(p.numel() for p in DeterministicConditional().parameters()),2611864)

if __name__=='__main__': unittest.main()
