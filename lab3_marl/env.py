"""Lab 3: cooperative multi-agent portfolio environment (batched TorchRL EnvBase, weekly steps).

Agents: 4 sector agents (each manages two assets) and, optionally, 1 risk agent (gross exposure).
Discrete action (9 values) — sector agent: scores (s1, s2) in {0,1,2}^2; risk agent: exposure 0.2 + 0.1 * a.
Continuous action ([-1, 1]^2) — the same quantities on a continuous scale (used by MADDPG / IDDPG).
Weights: w = exposure * score / sum(score); the rest is cash earning the 13-week T-bill yield known at the decision.
Reward (shared, in percent): 100 * [log portfolio return - cost_bps * turnover - lam_var * logret^2
                                    - lam_conc * sum(max(w - 0.25, 0))]; turnover is against drifted weights.
Episode: train mode — random start in the split, 52 weeks, stop early if the drawdown exceeds 25%;
eval mode — one pass over the whole split.
"""
import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import Bounded, Categorical, Composite, Unbounded
from torchrl.envs import EnvBase

N_SECTOR = 4
N_ACTIONS = 9
MAX_W = 0.25
DD_STOP = 0.25


class PortfolioEnv(EnvBase):
    batch_locked = True

    def __init__(self, panel, split='train', num_envs=16, episode_len=52, risk_agent=True, cost_bps=10.0,
                 reward_cost=True, lam_var=2.0, lam_conc=1.0, continuous=False, single=False, dd_stop=DD_STOP,
                 seed=0, device='cpu'):
        super().__init__(device=device, batch_size=torch.Size([num_envs]))
        mask = panel['split'] == split
        self.X = torch.tensor(panel['X'][mask], dtype=torch.float32, device=device)       # [T, A, F]
        self.R = torch.tensor(panel['R'][mask], dtype=torch.float32, device=device)       # [T, A] log
        self.C = torch.tensor(panel['C'][mask], dtype=torch.float32, device=device)       # [T] cash log return
        self.T, self.A, self.F = self.X.shape
        self.eval_mode = split != 'train'
        self.episode_len = self.T - 1 if self.eval_mode else episode_len
        self.risk_agent, self.cost, self.reward_cost = risk_agent, cost_bps / 1e4, reward_cost
        self.dd_stop = dd_stop                                              # None: no early stop on drawdown
        self.lam_var, self.lam_conc, self.continuous, self.single = lam_var, lam_conc, continuous or single, single
        self.n_agents = 1 if single else N_SECTOR + int(risk_agent)
        self.P = 4                                                          # exposure, drawdown, last return, cash yield
        self.state_dim = self.A * self.F + self.A + self.P
        self.obs_dim = self.state_dim + 1 if single else 2 * self.F + 2 + self.F + self.P + self.n_agents
        self.group_map = {'agents': [f'agent_{i}' for i in range(self.n_agents)]}
        self._make_specs()
        self.rng = torch.Generator(device='cpu').manual_seed(seed)

    # ---------- specs ----------
    def _make_specs(self):
        B, N = self.batch_size, self.n_agents
        self.observation_spec = Composite(
            agents=Composite(observation=Unbounded(shape=(*B, N, self.obs_dim)), shape=(*B, N)),
            state=Unbounded(shape=(*B, self.state_dim)), shape=B)
        if self.single:
            act = Bounded(low=-1.0, high=1.0, shape=(*B, N, 2 * N_SECTOR + 1))
        elif self.continuous:
            act = Bounded(low=-1.0, high=1.0, shape=(*B, N, 2))
        else:
            act = Categorical(n=N_ACTIONS, shape=(*B, N), dtype=torch.int64)
        self.action_spec = Composite(agents=Composite(action=act, shape=(*B, N)), shape=B)
        self.reward_spec = Composite(agents=Composite(reward=Unbounded(shape=(*B, N, 1)), shape=(*B, N)), shape=B)
        self.done_spec = Composite(done=Categorical(2, shape=(*B, 1), dtype=torch.bool),
                                   terminated=Categorical(2, shape=(*B, 1), dtype=torch.bool),
                                   truncated=Categorical(2, shape=(*B, 1), dtype=torch.bool), shape=B)

    # ---------- dynamics ----------
    def _set_seed(self, seed):
        self.rng = torch.Generator(device='cpu').manual_seed(int(seed) if seed is not None else 0)

    def _observe(self):
        x = self.X[self.t]                                                  # [B, A, F]
        market = x.mean(1)                                                  # [B, F]
        port = torch.stack([self.w.sum(1), self.dd, self.last_ret * 10, self.C[self.t] * 520], -1)  # [B, 4]
        eye = torch.eye(max(self.n_agents, N_SECTOR + int(self.risk_agent)), device=self.device)[:self.n_agents]
        obs = []
        for k in range(N_SECTOR):
            own = x[:, 2 * k:2 * k + 2].reshape(x.shape[0], -1)
            obs.append(torch.cat([own, self.w[:, 2 * k:2 * k + 2], market, port], -1))
        if self.risk_agent:
            disp = x.std(1)                                                 # cross-sectional dispersion [B, F]
            pad = torch.zeros(x.shape[0], (2 * self.F + 2 + self.F + self.P) - (2 * self.F + self.P + self.A),
                              device=self.device)
            risk = torch.cat([disp, market, port, self.w, pad], -1)       # same length as a sector observation
            obs.append(risk)
        obs = torch.stack(obs, 1)                                           # [B, N, obs_dim - N]
        if not self.single:
            obs = torch.cat([obs, eye.expand(obs.shape[0], -1, -1)], -1)
        state = torch.cat([x.reshape(x.shape[0], -1), self.w, port], -1)
        if self.single:
            obs = torch.cat([state, torch.ones(x.shape[0], 1, device=self.device)], -1).unsqueeze(1)
        return obs, state

    def _weights(self, action):
        B = action.shape[0]
        if self.single:
            a = action[:, 0].clamp(-1, 1)                                   # [B, 9]
            scores = a[:, :2 * N_SECTOR] + 1.0
            exposure = 0.2 + 0.4 * (a[:, 2 * N_SECTOR] + 1.0)
        elif self.continuous:
            a = action.clamp(-1, 1)
            scores = (a[:, :N_SECTOR] + 1.0).reshape(B, -1)                 # [B, 8] in [0, 2]
            exposure = 0.2 + 0.4 * (a[:, N_SECTOR, 0] + 1.0) if self.risk_agent else torch.ones(B, device=self.device)
        else:
            a = action.long()
            s1, s2 = a[:, :N_SECTOR] // 3, a[:, :N_SECTOR] % 3
            scores = torch.stack([s1, s2], -1).reshape(B, -1).float()
            exposure = 0.2 + 0.1 * a[:, N_SECTOR].float() if self.risk_agent else torch.ones(B, device=self.device)
        total = scores.sum(1, keepdim=True)
        w = torch.where(total > 0, scores / total.clamp_min(1e-9), torch.zeros_like(scores))
        return w * exposure.unsqueeze(1)

    def _reset(self, tensordict=None, **kwargs):
        B = self.batch_size[0]
        if tensordict is not None and '_reset' in tensordict.keys():
            mask = tensordict['_reset'].reshape(B)
        else:
            mask = torch.ones(B, dtype=torch.bool, device=self.device)
        if not hasattr(self, 't'):
            self.t = torch.zeros(B, dtype=torch.long, device=self.device)
            self.step_count = torch.zeros(B, dtype=torch.long, device=self.device)
            self.w = torch.zeros(B, self.A, device=self.device)
            self.value = torch.ones(B, device=self.device)
            self.peak = torch.ones(B, device=self.device)
            self.dd = torch.zeros(B, device=self.device)
            self.last_ret = torch.zeros(B, device=self.device)
        if self.eval_mode:
            start = torch.zeros(B, dtype=torch.long)
        else:
            start = torch.randint(0, self.T - self.episode_len - 1, (B,), generator=self.rng)
        start = start.to(self.device)
        self.t = torch.where(mask, start, self.t)
        self.step_count = torch.where(mask, torch.zeros_like(self.step_count), self.step_count)
        m = mask.unsqueeze(1)
        self.w = torch.where(m, torch.full_like(self.w, 1.0 / self.A), self.w)     # start from equal weights
        for name in ('value', 'peak'):
            setattr(self, name, torch.where(mask, torch.ones_like(self.value), getattr(self, name)))
        self.dd = torch.where(mask, torch.zeros_like(self.dd), self.dd)
        self.last_ret = torch.where(mask, torch.zeros_like(self.last_ret), self.last_ret)
        obs, state = self._observe()
        false = torch.zeros(B, 1, dtype=torch.bool, device=self.device)
        return TensorDict({'agents': TensorDict({'observation': obs}, batch_size=(B, self.n_agents)), 'state': state,
                           'done': false, 'terminated': false.clone(), 'truncated': false.clone()}, batch_size=(B,))

    def _step(self, tensordict):
        B = self.batch_size[0]
        action = tensordict['agents', 'action']
        w_new = self._weights(action)
        r = self.R[self.t]                                                  # [B, A] next-week log returns
        gross = torch.exp(r) - 1.0
        cash = torch.expm1(self.C[self.t])                                  # [B] cash return over the week
        # drift of the previous weights over the past week is already applied in self.w
        turnover = (w_new - self.w).abs().sum(1)
        cost = self.cost * turnover
        port_simple = (w_new * gross).sum(1) + (1.0 - w_new.sum(1)) * cash
        logret = torch.log1p(port_simple) - cost
        excess = (w_new - MAX_W).clamp_min(0).sum(1)
        pen = self.lam_var * logret ** 2 + self.lam_conc * excess
        reward = 100.0 * (logret - pen) if self.reward_cost else 100.0 * (torch.log1p(port_simple) - pen)
        # bookkeeping: value, drawdown, drifted weights for the next decision
        self.value = self.value * torch.exp(logret)
        self.peak = torch.maximum(self.peak, self.value)
        self.dd = 1.0 - self.value / self.peak
        grown = w_new * (1.0 + gross)
        self.w = grown / (1.0 + port_simple).unsqueeze(1)
        self.last_ret = logret
        self.t = self.t + 1
        self.step_count = self.step_count + 1
        self.info = {'turnover': turnover, 'cost': cost, 'logret': logret, 'excess': excess, 'w': w_new,
                     'cash': torch.log1p(cash),
                     'violation': (w_new > MAX_W + 1e-9).any(1)}
        if self.eval_mode or self.dd_stop is None:
            terminated = torch.zeros(B, 1, dtype=torch.bool)
        else:
            terminated = (self.dd > self.dd_stop).unsqueeze(1)
        truncated = (self.step_count >= self.episode_len).unsqueeze(1)
        done = terminated | truncated
        self.t = torch.clamp(self.t, max=self.T - 1)
        obs, state = self._observe()
        rew = reward.reshape(B, 1, 1).expand(B, self.n_agents, 1).clone()
        return TensorDict({'agents': TensorDict({'observation': obs, 'reward': rew}, batch_size=(B, self.n_agents)),
                           'state': state, 'done': done, 'terminated': terminated.to(self.device),
                           'truncated': truncated}, batch_size=(B,))


def load_panel(name):
    import pathlib
    data = np.load(pathlib.Path(__file__).resolve().parent / 'data' / f'panel_{name}.npz', allow_pickle=True)  # our own file written by features.py
    return {k: data[k] for k in data.files}
