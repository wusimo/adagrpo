#!/usr/bin/env python3
"""Debug: Why is grad_norm = 0 in the GRPO update?"""

import torch
import torch.nn as nn
import numpy as np

DEVICE = torch.device("cuda")

# Inline the minimal components
class SinusoidalEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half, device=t.device).float() / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

class NoiseNet(nn.Module):
    def __init__(self, action_dim=2, obs_dim=4, hidden=64):
        super().__init__()
        self.t_emb = SinusoidalEmbed(hidden)
        self.obs_proj = nn.Linear(obs_dim, hidden)
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden * 2, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, action_dim),
        )
    def forward(self, x_t, t, obs):
        te = self.t_emb(t)
        oe = self.obs_proj(obs)
        return self.net(torch.cat([x_t, te, oe], dim=-1))

torch.manual_seed(42)

net = NoiseNet().to(DEVICE)
obs = torch.randn(4, 4, device=DEVICE)
a_k = torch.randn(4, 2, device=DEVICE)  # Detached input (from rollout)
t = torch.full((4,), 25, device=DEVICE, dtype=torch.long)

# 1. Compute mean through the network (should be differentiable)
noise_pred = net(a_k, t, obs)
print(f"noise_pred requires_grad: {noise_pred.requires_grad}")
print(f"noise_pred grad_fn: {noise_pred.grad_fn}")

# Simulate DDIM mean computation
ac_t = torch.tensor(0.8, device=DEVICE)
ac_prev = torch.tensor(0.9, device=DEVICE)
pred_x0 = (a_k - (1 - ac_t).sqrt() * noise_pred) / ac_t.sqrt()
dir_xt = (1 - ac_prev).clamp(min=0).sqrt() * noise_pred
mu_new = ac_prev.sqrt() * pred_x0 + dir_xt
print(f"mu_new requires_grad: {mu_new.requires_grad}")

# 2. Compute the ratio
mu_old = mu_new.detach() + 0.01  # Slight offset to ensure ratio != 1
a_km1 = torch.randn(4, 2, device=DEVICE)  # Detached sample
sigma = 0.3

inv_2sig2 = 0.5 / (sigma**2)
diff_old = (a_km1 - mu_old).pow(2).sum(dim=-1)
diff_new = (a_km1 - mu_new).pow(2).sum(dim=-1)
log_r = inv_2sig2 * (diff_old - diff_new)
print(f"\nlog_r: {log_r}")
print(f"log_r requires_grad: {log_r.requires_grad}")
print(f"log_r grad_fn: {log_r.grad_fn}")

ratio = torch.exp(log_r.clamp(-10, 10))
print(f"ratio: {ratio}")

# 3. Compute loss
advantages = torch.tensor([1.0, -1.0, 0.5, -0.5], device=DEVICE)
loss = -(ratio * advantages).mean()
print(f"\nloss: {loss.item()}")
print(f"loss requires_grad: {loss.requires_grad}")

# 4. Backward
loss.backward()

grad_norm = sum(p.grad.norm().item()**2 for p in net.parameters() if p.grad is not None)**0.5
print(f"grad_norm: {grad_norm}")

if grad_norm > 0:
    print("\n✓ GRADIENT FLOWS! The issue was elsewhere.")
else:
    print("\n✗ Gradient is zero — fundamental problem.")

# 5. Now test the ACTUAL issue: when mu_old == mu_new (same network weights)
print("\n--- Same-weight test (ratio = 1 exactly) ---")
net.zero_grad()

# Same network for both
noise_pred_2 = net(a_k, t, obs)
ac_t2 = torch.tensor(0.8, device=DEVICE)
ac_prev2 = torch.tensor(0.9, device=DEVICE)
pred_x0_2 = (a_k - (1 - ac_t2).sqrt() * noise_pred_2) / ac_t2.sqrt()
dir_xt_2 = (1 - ac_prev2).clamp(min=0).sqrt() * noise_pred_2
mu_new_2 = ac_prev2.sqrt() * pred_x0_2 + dir_xt_2

# mu_old is the SAME computation but detached
mu_old_2 = mu_new_2.detach()

diff_old_2 = (a_km1 - mu_old_2).pow(2).sum(dim=-1)
diff_new_2 = (a_km1 - mu_new_2).pow(2).sum(dim=-1)
log_r_2 = inv_2sig2 * (diff_old_2 - diff_new_2)
ratio_2 = torch.exp(log_r_2.clamp(-10, 10))
print(f"log_r when same weights: {log_r_2}")
print(f"ratio when same weights: {ratio_2}")

loss_2 = -(ratio_2 * advantages).mean()
loss_2.backward()

grad_norm_2 = sum(p.grad.norm().item()**2 for p in net.parameters() if p.grad is not None)**0.5
print(f"grad_norm when same weights: {grad_norm_2}")

if grad_norm_2 > 0:
    print("✓ Gradient flows even when ratio=1 (d/dθ ratio ≠ 0 at θ=θ_old)")
else:
    print("✗ Gradient zero when ratio=1 — this IS the fundamental problem")
    print("  ratio = exp(log_r), log_r = c*(||a-mu_old||² - ||a-mu_new||²)")
    print("  When mu_old = mu_new: log_r = 0, ratio = 1")
    print("  d(ratio)/d(mu_new) = ratio * d(log_r)/d(mu_new)")
    print("  d(log_r)/d(mu_new) = c * 2 * (a - mu_new) * d(mu_new)/d(θ)")
    print("  This should be NON-ZERO even when mu_old = mu_new!")
    print("  Unless the computation graph is somehow broken...")
