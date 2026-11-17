"""Historical BBDM-inspired bridge, retained to document a closed branch.

Forward marginals use delta=2*s*m*(1-m), tau=x_t-x0. The exact terminal
entrance is stochastic, followed by deterministic interior subsection
jumps. This endpoint extension is not a claim of author-code reproduction.
"""
from __future__ import annotations
import math
from numbers import Integral
import torch

class BrownianBridge:
    def __init__(self, timesteps: int = 1000, s: float = 1.0):
        if isinstance(timesteps, bool) or not isinstance(timesteps, Integral) or timesteps < 1:
            raise ValueError('timesteps must be a positive integer')
        if not math.isfinite(float(s)) or float(s) <= 0:
            raise ValueError('s must be finite and positive')
        self.T = int(timesteps)
        self.s = float(s)
        self.m = torch.arange(self.T + 1, dtype=torch.float64) / self.T
        self.delta = 2.0 * self.s * self.m * (1.0 - self.m)

    @staticmethod
    def _pair(x, y):
        if (not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor)
                or not x.is_floating_point() or not y.is_floating_point()
                or x.ndim < 2 or x.shape != y.shape or x.shape[0] < 1
                or x.device != y.device or x.dtype != y.dtype):
            raise ValueError('states must be matching floating tensors with a nonempty batch')

    def _time(self, t, x, minimum=0):
        tv = torch.as_tensor(t, device=x.device)
        if tv.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError('time indices must have integer dtype')
        if tv.ndim == 0:
            tv = tv.reshape(1)
        if tv.ndim != 1 or tv.numel() not in (1, x.shape[0]):
            raise ValueError('time indices must be scalar or one per batch item')
        tv = tv.long().expand(x.shape[0])
        if bool(((tv < minimum) | (tv > self.T)).any()):
            raise ValueError(f'time indices must lie in [{minimum}, {self.T}]')
        return tv

    @staticmethod
    def _b(value, x):
        return value.to(device=x.device, dtype=x.dtype).reshape(-1, *([1] * (x.ndim - 1)))

    def _at(self, arr, t, x):
        return self._b(arr.to(x.device)[t], x)

    @staticmethod
    def _noise(x, noise=None, generator=None):
        if noise is not None:
            BrownianBridge._pair(x, noise)
            return noise
        return torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)

    @staticmethod
    def _scalar_time(t):
        if isinstance(t, bool) or not isinstance(t, Integral):
            raise ValueError('sampling jump indices must be integers')
        return int(t)

    def q_sample(self, x0, y, t, noise=None, *, generator=None):
        """Marginal forward sample, including the exact t=0 and t=T endpoints."""
        self._pair(x0, y)
        tv = self._time(t, x0)
        mt, dt = self._at(self.m, tv, x0), self._at(self.delta, tv, x0)
        eps = self._noise(x0, noise, generator)
        xt = (1.0 - mt) * x0 + mt * y + torch.sqrt(dt) * eps
        return xt, xt - x0

    def chain_step(self, x_prev, y, t, noise=None, *, generator=None):
        """Actual Markov q(x_t|x_(t-1),y), with scalar or batched t in 1..T."""
        self._pair(x_prev, y)
        tv = self._time(t, x_prev, minimum=1)
        mt, mp = self._at(self.m, tv, x_prev), self._at(self.m, tv - 1, x_prev)
        a = (1.0 - mt) / (1.0 - mp)
        variance = 2.0 * self.s * (1.0 - mt) * (mt - mp) / (1.0 - mp)
        eps = self._noise(x_prev, noise, generator)
        return a * x_prev + (mt - a * mp) * y + torch.sqrt(variance) * eps

    def posterior(self, xt, x0, y, t):
        """Exact adjacent-step conditional mean/variance, including t=T.

        At t=1 returns exactly (x0,0). At t=T the observed xt must equal y.
        """
        self._pair(xt, x0)
        self._pair(xt, y)
        tv = self._time(t, xt, minimum=1)
        terminal = tv == self.T
        if bool(terminal.any()) and not torch.equal(xt[terminal], y[terminal]):
            raise ValueError('the exact terminal state x_T must equal y')
        mt, mp = self._at(self.m, tv, xt), self._at(self.m, tv - 1, xt)
        weight = (mt - mp) / mt
        mean = (1.0 - weight) * xt + weight * x0
        variance = 2.0 * self.s * mp * (mt - mp) / mt
        return mean, variance

    def posterior_step(self, xt, tau_hat, y, t, noise=None, *, generator=None):
        """Stochastic Markov reverse sample. Supply zero noise to request its mean."""
        self._pair(xt, tau_hat)
        mean, var = self.posterior(xt, xt - tau_hat, y, t)
        if bool((var == 0).all()):
            return mean
        return mean + torch.sqrt(var) * self._noise(xt, noise, generator)

    def ddim_step(self, xt, tau_hat, y, t_cur: int, t_prev: int, noise=None, *, generator=None):
        """Exact-T stochastic entrance; sigma=0 subsection updates thereafter.

        Omitted entrance noise creates a fresh Gaussian, not a zero-noise mean.
        This matches oracle intermediate training marginals at the first jump.
        """
        self._pair(xt, tau_hat)
        self._pair(xt, y)
        tc, tp = self._scalar_time(t_cur), self._scalar_time(t_prev)
        if not 0 <= tp < tc <= self.T:
            raise ValueError('jump must satisfy 0 <= t_prev < t_cur <= T')
        x0_hat = xt - tau_hat
        if tp == 0:
            return x0_hat
        mc, mp = float(self.m[tc]), float(self.m[tp])
        dc, dp = float(self.delta[tc]), float(self.delta[tp])
        if tc == self.T:
            if not torch.equal(xt, y):
                raise ValueError('the exact terminal state x_T must equal y')
            return ((1.0 - mp) * x0_hat + mp * y
                    + math.sqrt(dp) * self._noise(xt, noise, generator))
        if noise is not None:
            raise ValueError('sigma=0 interior jumps do not accept a noise tensor')
        return ((1.0 - mp) * x0_hat + mp * y + math.sqrt(dp / dc)
                * (xt - (1.0 - mc) * x0_hat - mc * y))

    def sampling_times(self, steps: int):
        if isinstance(steps, bool) or not isinstance(steps, Integral) or not 1 <= steps <= self.T:
            raise ValueError('steps must be an integer in [1,T]')
        seq = torch.linspace(self.T, 0, int(steps) + 1, dtype=torch.float64).round().long()
        if not bool((seq[:-1] > seq[1:]).all()):
            raise RuntimeError('sampling schedule has repeated or unordered indices')
        return seq.tolist()

    @torch.no_grad()
    def sample(self, y, tau_fn, steps: int = 10, noise_fn=None, *, generator=None):
        """Start at x_T=y; tau_fn(xt,t_batch) returns state-to-clean displacement.

        noise_fn(shape,k) is called only for the stochastic entrance (k=0).
        Alternatively pass a seeded torch.Generator. These choices are exclusive.
        One-jump T->0 inference has no intermediate stochastic state.
        """
        self._pair(y, y)
        if noise_fn is not None and generator is not None:
            raise ValueError('supply noise_fn or generator, not both')
        seq = self.sampling_times(steps)
        xt = y.clone()
        for k, (tc, tp) in enumerate(zip(seq[:-1], seq[1:])):
            tv = torch.full((y.shape[0],), tc, dtype=torch.long, device=y.device)
            tau_hat = tau_fn(xt, tv)
            noise = noise_fn(xt.shape, k) if noise_fn is not None and tc == self.T and tp > 0 else None
            xt = self.ddim_step(xt, tau_hat, y, tc, tp, noise, generator=generator)
        return xt


def bridge_loss(bb, net, x0, y, t, noise=None):
    """Unweighted tau-MSE as in the paper's practical Algorithm 1."""
    xt, tau = bb.q_sample(x0, y, t, noise)
    return torch.mean((net(xt, t) - tau) ** 2)
