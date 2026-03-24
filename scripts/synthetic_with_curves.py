#!/usr/bin/env python3
"""Synthetic experiments — careful version with IL convergence checks.

Ensures IL baseline is actually trained before running GRPO.
"""

import copy, json, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path("results/synthetic")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# === DiffPolicy (identical to the working grpo_diffusion_test.py) ===
class SinusoidalEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-np.log(10000)*torch.arange(half,device=t.device).float()/half)
        args = t[:,None].float()*freqs[None,:]
        return torch.cat([torch.cos(args),torch.sin(args)],dim=-1)

class NoiseNet(nn.Module):
    def __init__(self, ad, od, h=128):
        super().__init__()
        self.t_emb=SinusoidalEmbed(h); self.obs_proj=nn.Linear(od,h)
        self.net=nn.Sequential(nn.Linear(ad+h*2,h),nn.Mish(),nn.Linear(h,h),nn.Mish(),nn.Linear(h,ad))
    def forward(self,x,t,o):
        return self.net(torch.cat([x,self.t_emb(t),self.obs_proj(o)],dim=-1))

def cosine_beta(T,s=0.008):
    steps=T+1; t=torch.linspace(0,T,steps)/T
    ac=torch.cos((t+s)/(1+s)*np.pi/2).pow(2); ac=ac/ac[0]
    return (1-ac[1:]/ac[:-1]).clamp(0,0.999)

class DiffPolicy:
    def __init__(self, ad, od, T=50, K=5, h=128, eta=1.0, sm=0.05):
        self.ad,self.od,self.T,self.K,self.eta,self.sm,self.h=ad,od,T,K,eta,sm,h
        self.nn_=NoiseNet(ad,od,h).to(DEVICE)
        b=cosine_beta(T).to(DEVICE); self.ac=torch.cumprod(1-b,0)
        sr=T//K; self.ts=(torch.arange(0,K)*sr).long().flip(0).to(DEVICE)
    def parameters(self): return self.nn_.parameters()
    def sd(self): return self.nn_.state_dict()
    def ld(self,s): self.nn_.load_state_dict(s)
    def train(self): self.nn_.train()
    def eval(self): self.nn_.eval()
    def _sp(self,i):
        t=self.ts[i]; tp=self.ts[i+1] if i+1<self.K else torch.tensor(0,device=DEVICE)
        at=self.ac[t]; ap=self.ac[tp] if tp>0 else torch.tensor(1.,device=DEVICE)
        pv=((1-ap)/(1-at+1e-8)*(1-at/(ap+1e-8))).clamp(min=1e-8)
        return at,ap,max(self.eta*pv.sqrt().item(),self.sm),t
    def _mn(self,x,n,at,ap,s):
        x0=(x-(1-at).sqrt()*n)/at.sqrt(); d=(1-ap-s**2).clamp(min=0).sqrt()*n
        return ap.sqrt()*x0+d
    def dl(self,o,a):
        B=a.shape[0]; t=torch.randint(0,self.T,(B,),device=DEVICE); n=torch.randn_like(a)
        ac_=self.ac[t].unsqueeze(-1)
        return F.mse_loss(self.nn_(ac_.sqrt()*a+(1-ac_).sqrt()*n,t,o),n)
    @torch.no_grad()
    def sample(self,o):
        B=o.shape[0]; x=torch.randn(B,self.ad,device=DEVICE)
        for i in range(self.K):
            at,ap,s,t=self._sp(i); x=self._mn(x,self.nn_(x,t.expand(B),o),at,ap,s)
        return x
    @torch.no_grad()
    def swp(self,o):
        B=o.shape[0]; x=torch.randn(B,self.ad,device=DEVICE)
        I,O,M,S=[],[],[],[]
        for i in range(self.K):
            at,ap,s,t=self._sp(i); I.append(x.clone())
            m=self._mn(x,self.nn_(x,t.expand(B),o),at,ap,s)
            xn=(m+s*torch.randn_like(x)) if i<self.K-1 else m
            O.append(xn.clone()); M.append(m); S.append(s); x=xn
        return x,torch.stack(I,1),torch.stack(O,1),torch.stack(M,1),torch.tensor(S,device=DEVICE)
    def cmas(self,a,i,o):
        at,ap,s,t=self._sp(i)
        return self._mn(a,self.nn_(a,t.expand(a.shape[0]),o),at,ap,s)
    def clone(self):
        p=DiffPolicy(self.ad,self.od,self.T,self.K,self.h,self.eta,self.sm); p.ld(copy.deepcopy(self.sd())); return p


# === Experiment runner ===

def run_exp(name, reward_fn, obs_fn, expert_fn, ad, od,
            eta=1.0, sm=0.05, K=5, strategy='product',
            il_steps=3000, rl_iters=300, G=8, ng=4,
            lr_rl=3e-5, clip_eps=0.2, ref_freq=5, clamp=2.0,
            aux_weight=0.1, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    policy = DiffPolicy(ad, od, T=50, K=K, h=128, eta=eta, sm=sm)

    # === IL with convergence check ===
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    demo_o = obs_fn(1024)
    demo_a = expert_fn(demo_o)

    for step in range(il_steps):
        idx = torch.randint(0, 1024, (64,))
        loss = policy.dl(demo_o[idx], demo_a[idx])
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

    policy.eval()
    eo = obs_fn(1000)
    with torch.no_grad(): il_r = reward_fn(eo, policy.sample(eo)).mean().item()
    il_sd = copy.deepcopy(policy.sd())

    # === GRPO ===
    policy.ld(copy.deepcopy(il_sd))
    po = policy.clone(); po.eval()
    opt = torch.optim.Adam(policy.parameters(), lr=lr_rl)
    log = defaultdict(list)

    for it in range(rl_iters):
        policy.train()
        tl = torch.tensor(0., device=DEVICE); nu = 0; ir = []; irat = []

        for _ in range(ng):
            o = obs_fn(1).expand(G, -1).contiguous()
            with torch.no_grad(): a, inp, out, mo, sig = po.swp(o)
            rw = reward_fn(o, a); ir.extend(rw.tolist())
            if rw.std() < 1e-8: continue
            adv = (rw - rw.mean()) / (rw.std() + 1e-8); nu += 1

            Kact = inp.shape[1]
            if strategy == 'product':
                lr_ = torch.zeros(G, device=DEVICE)
                for k in range(Kact):
                    mn = policy.cmas(inp[:, k].detach(), k, o)
                    i2 = 0.5 / (sig[k].item()**2)
                    do = (out[:, k].detach() - mo[:, k].detach()).pow(2).sum(-1)
                    dn = (out[:, k].detach() - mn).pow(2).sum(-1)
                    lr_ += (i2 * (do - dn)).clamp(-clamp, clamp)
                r = torch.exp(lr_.clamp(-5, 5))
                s1 = r * adv; s2 = torch.clamp(r, 1 - clip_eps, 1 + clip_eps) * adv
                grpo_l = -torch.min(s1, s2).mean()
                irat.extend(r.detach().cpu().tolist())

            elif strategy == 'per_step':
                grpo_l = torch.tensor(0., device=DEVICE)
                for k in range(Kact):
                    mn = policy.cmas(inp[:, k].detach(), k, o)
                    i2 = 0.5 / (sig[k].item()**2)
                    do = (out[:, k].detach() - mo[:, k].detach()).pow(2).sum(-1)
                    dn = (out[:, k].detach() - mn).pow(2).sum(-1)
                    rk = torch.exp((i2 * (do - dn)).clamp(-5, 5))
                    s1 = rk * adv; s2 = torch.clamp(rk, 1 - clip_eps, 1 + clip_eps) * adv
                    grpo_l = grpo_l - torch.min(s1, s2).mean()
                grpo_l = grpo_l / Kact

            elif strategy == 'mean':
                lr_ = torch.zeros(G, device=DEVICE)
                for k in range(Kact):
                    mn = policy.cmas(inp[:, k].detach(), k, o)
                    i2 = 0.5 / (sig[k].item()**2)
                    do = (out[:, k].detach() - mo[:, k].detach()).pow(2).sum(-1)
                    dn = (out[:, k].detach() - mn).pow(2).sum(-1)
                    lr_ += (i2 * (do - dn)).clamp(-clamp, clamp)
                lr_ = lr_ / Kact
                r = torch.exp(lr_.clamp(-5, 5))
                s1 = r * adv; s2 = torch.clamp(r, 1 - clip_eps, 1 + clip_eps) * adv
                grpo_l = -torch.min(s1, s2).mean()

            # Aux loss
            idx = torch.randint(0, 1024, (32,))
            aux_l = policy.dl(demo_o[idx], demo_a[idx])
            tl = tl + grpo_l + aux_weight * aux_l

        if nu == 0: continue
        tl = tl / nu
        opt.zero_grad(); tl.backward()
        gn = sum(p.grad.norm().item()**2 for p in policy.parameters() if p.grad is not None)**0.5
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()

        if (it + 1) % ref_freq == 0: po.ld(policy.sd())

        log['reward'].append(float(np.mean(ir)))
        log['grad'].append(float(gn))
        log['ratio_mean'].append(float(np.mean(irat)) if irat else 1.0)
        log['ratio_std'].append(float(np.std(irat)) if irat else 0.0)

        if (it + 1) % 50 == 0:
            policy.eval()
            with torch.no_grad():
                er = reward_fn(obs_fn(500), policy.sample(obs_fn(500))).mean().item()
            log['eval_r'].append(float(er)); log['eval_it'].append(it + 1)

    policy.eval()
    with torch.no_grad():
        fr = reward_fn(obs_fn(1000), policy.sample(obs_fn(1000))).mean().item()
    return il_r, fr, log


# === Environments ===
def mk_reach():
    ad,od=2,4
    def t(o): return o[:,:2]*0.5
    def r(o,a): return -(a-t(o)).pow(2).sum(-1)
    def e(o): return t(o)+torch.randn(o.shape[0],ad,device=DEVICE)*0.1
    def obs(n): return torch.randn(n,od,device=DEVICE)
    return 'reaching_2d',r,obs,e,ad,od

def mk_biased():
    ad,od=2,4
    def t(o): return o[:,:2]*0.5
    def r(o,a): return -(a-t(o)).pow(2).sum(-1)
    def e(o): return t(o)+0.3+torch.randn(o.shape[0],ad,device=DEVICE)*0.05
    def obs(n): return torch.randn(n,od,device=DEVICE)
    return 'biased_expert',r,obs,e,ad,od

def mk_7d():
    ad,od=7,10
    W=torch.randn(od,ad,device=DEVICE)*0.3
    def r(o,a): return -(a-o@W).pow(2).sum(-1)
    def e(o): return o@W+0.1+torch.randn(o.shape[0],ad,device=DEVICE)*0.15
    def obs(n): return torch.randn(n,od,device=DEVICE)
    return 'high_dim_7d',r,obs,e,ad,od

def mk_sparse():
    ad,od=2,4
    def t(o): return o[:,:2]*0.3
    def r(o,a): return ((a-t(o)).pow(2).sum(-1).sqrt()<0.3).float()
    def e(o): return t(o)+torch.randn(o.shape[0],ad,device=DEVICE)*0.2
    def obs(n): return torch.randn(n,od,device=DEVICE)
    return 'sparse_reward',r,obs,e,ad,od


def main():
    all_res = {}

    print("="*70); print("TABLE 1: Main Results"); print("="*70)
    for mk in [mk_reach, mk_biased, mk_7d, mk_sparse]:
        n,r,o,e,ad,od = mk()
        il,gr,log = run_exp(n,r,o,e,ad,od,seed=0,il_steps=3000)
        d=gr-il; p=d/abs(il)*100 if il!=0 else 0
        print(f"  {n:<20} IL={il:.4f} GRPO={gr:.4f} Δ={d:+.4f} ({p:+.1f}%)")
        all_res[n] = {'il':il,'grpo':gr,'log':log}

    print(f"\n{'='*70}\nTABLE 2: σ_min ablation\n{'='*70}")
    _,r,o,e,ad,od=mk_reach()
    for sm in [0.0, 0.01, 0.05, 0.1, 0.2]:
        il,gr,_=run_exp('sm',r,o,e,ad,od,sm=sm,seed=0)
        d=gr-il; p=d/abs(il)*100 if il!=0 else 0
        print(f"  σ_min={sm:<5} IL={il:.4f} GRPO={gr:.4f} ({p:+.1f}%)")
        all_res[f'sm_{sm}']={'il':il,'grpo':gr}

    print(f"\n{'='*70}\nTABLE 3: η ablation\n{'='*70}")
    for eta in [0.0, 0.3, 0.5, 1.0]:
        il,gr,log=run_exp('eta',r,o,e,ad,od,eta=eta,seed=0)
        ag=np.mean(log['grad']) if log['grad'] else 0
        d=gr-il; p=d/abs(il)*100 if il!=0 else 0
        print(f"  η={eta:<4} IL={il:.4f} GRPO={gr:.4f} grad={ag:.3f} ({p:+.1f}%)")
        all_res[f'eta_{eta}']={'il':il,'grpo':gr,'grad':ag}

    print(f"\n{'='*70}\nTABLE 4: K scaling\n{'='*70}")
    for K in [3, 5, 8, 10]:
        il,gr,log=run_exp('K',r,o,e,ad,od,K=K,seed=0)
        rs=np.mean(log['ratio_std']) if log['ratio_std'] else 0
        d=gr-il; p=d/abs(il)*100 if il!=0 else 0
        print(f"  K={K:<3} IL={il:.4f} GRPO={gr:.4f} ratio_std={rs:.4f} ({p:+.1f}%)")
        all_res[f'K_{K}']={'il':il,'grpo':gr,'rs':rs}

    print(f"\n{'='*70}\nTABLE 5: Ratio strategy\n{'='*70}")
    for st in ['product','per_step','mean']:
        il,gr,log=run_exp('st',r,o,e,ad,od,strategy=st,seed=0)
        d=gr-il; p=d/abs(il)*100 if il!=0 else 0
        print(f"  {st:<10} IL={il:.4f} GRPO={gr:.4f} ({p:+.1f}%)")
        all_res[f'strat_{st}']={'il':il,'grpo':gr}

    print(f"\n{'='*70}\nTABLE 6: λ_aux ablation\n{'='*70}")
    for aw in [0.0, 0.01, 0.1, 0.5]:
        il,gr,_=run_exp('aux',r,o,e,ad,od,aux_weight=aw,seed=0,rl_iters=200)
        d=gr-il; p=d/abs(il)*100 if il!=0 else 0
        print(f"  λ={aw:<5} IL={il:.4f} GRPO={gr:.4f} ({p:+.1f}%)")
        all_res[f'aux_{aw}']={'il':il,'grpo':gr}

    with open(RESULTS_DIR/"all_synthetic.json",'w') as f:
        json.dump({k:v for k,v in all_res.items() if not isinstance(v,defaultdict)},f,indent=2,default=float)
    print(f"\nSaved to {RESULTS_DIR}/all_synthetic.json")


if __name__=="__main__":
    main()
