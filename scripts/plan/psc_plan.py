"""Planification PSC (10/10/2026) : contexte de H frames, optimiseur à gradient, oracle.

stable-worldmodel 0.1.1 ne donne au planificateur qu'UNE frame de contexte : pour
Pendulum / MountainCar, la vitesse de départ est alors inconnue (une image seule ne
la contient pas). Ce module ajoute, sans toucher à la lib :

- rollout_with_history : rollout de LeWM (et de PhaseSpaceLeWM, même interface
  predict) qui prend H frames de contexte ET les H-1 blocs d'actions déjà exécutés
  entre elles (info['action_history']) ; toutes les actions candidates sont futures.
  Repris du rollout de stable-worldmodel HEAD. Avec H = 1, identique à 0.1.1.
- HistoryPolicy : WorldModelPolicy qui fournit au solveur les H dernières frames
  (une tous les `action_block` pas) et les blocs d'actions entre elles. Au départ,
  ce passé vient du dataset (la trajectoire qui mène à l'état de départ).
- BoundedGradientSolver : GradientSolver de la lib (Adam sur la séquence
  d'actions, à travers le modèle), avec les actions ramenées dans leurs bornes
  (en unités normalisées) après chaque pas.
- OraclePolicy : même problème, résolu par CEM sur le VRAI simulateur (coût =
  erreur de position au pas de vérification) : plafond du protocole.
"""

import copy
from collections import deque

import numpy as np
import torch
from einops import rearrange

import stable_worldmodel as swm
from stable_worldmodel.solver import GradientSolver


# --------------------------------------------------------------- rollout
def rollout_with_history(self, info, action_sequence, history_size=None):
    """info['pixels'] : (B, S, H, C, h, w) ; action_sequence : (B, S, T, A) futures ;
    info['action_history'] : (B, S, H-1, A) blocs exécutés entre les frames de contexte.
    Retourne info avec predicted_emb (B, S, H + T, D)."""
    if history_size is None:
        history_size = getattr(self.predictor, 'num_frames', 3)
    H = info['pixels'].size(2)
    B, S, T = action_sequence.shape[:3]
    act_past = info.get('action_history')
    if act_past is None or H == 1:
        act_past = action_sequence.new_zeros(B, S, 0, action_sequence.size(-1))
    act_past = act_past.to(action_sequence.dtype)
    info['action'] = torch.cat([act_past, action_sequence[:, :, :1]], dim=2)

    if 'emb' not in info:
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        info['emb'] = _init['emb'].detach().unsqueeze(1).expand(B, S, -1, -1)

    emb_init = rearrange(info['emb'], 'b s ... -> (b s) ...')
    past = rearrange(act_past, 'b s ... -> (b s) ...')
    cand = rearrange(action_sequence, 'b s ... -> (b s) ...')
    all_act_emb = self.action_encoder(torch.cat([past, cand], dim=1))

    emb_list = list(emb_init.unbind(dim=1))
    for t in range(T):
        lo = max(0, H + t - history_size)
        emb_trunc = torch.stack(emb_list[lo:], dim=1)
        act_trunc = all_act_emb[:, lo:H + t]
        emb_list.append(self.predict(emb_trunc, act_trunc)[:, -1])
    emb = torch.stack(emb_list, dim=1)
    info['predicted_emb'] = rearrange(emb, '(b s) ... -> b s ...', b=B, s=S)
    return info


def install_history_rollout(model):
    model.rollout = rollout_with_history.__get__(model)
    return model


# --------------------------------------------------------------- solveur
class BoundedGradientSolver(GradientSolver):
    """GradientSolver avec actions bornées : après chaque pas d'optimisation, les
    actions (normalisées) sont ramenées dans [low, high] (bornes de l'environnement
    passées dans l'espace normalisé). Sans ça, le gradient peut pousser les actions
    hors bornes : l'environnement les écrête, le modèle non."""

    def __init__(self, *args, low=None, high=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.low, self.high = low, high

    def set_bounds(self, low, high):
        self.low, self.high = float(low), float(high)

    def init_action(self, n_envs, actions=None):
        # 0.1.1 laisse sur CPU un plan de depart (warm start) qui couvre deja tout
        # l'horizon, puis lui ajoute un bruit tire sur le GPU -> erreur de device
        if actions is not None:
            actions = actions.to(device=self.device, dtype=self.dtype)
        return super().init_action(n_envs, actions)

    def solve(self, info_dict, init_action=None):
        if self.low is None:
            return super().solve(info_dict, init_action)
        opt_cls, low, high = self.optimizer_cls, self.low, self.high

        class _Clamped(opt_cls):
            def step(self, *a, **k):
                out = super().step(*a, **k)
                with torch.no_grad():
                    for g in self.param_groups:
                        for p in g['params']:
                            p.clamp_(low, high)
                return out

        self.optimizer_cls = _Clamped
        try:
            return super().solve(info_dict, init_action)
        finally:
            self.optimizer_cls = opt_cls


# --------------------------------------------------------------- politiques
class HistoryPolicy(swm.policy.WorldModelPolicy):
    """Donne au solveur H frames de contexte (espacées de action_block pas) et
    les H-1 blocs d'actions exécutés entre elles. Passé initial : set_initial_history."""

    def __init__(self, *args, context=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.context = int(context)
        self._frames = None   # par env : deque de frames brutes (h, w, c)
        self._acts = None     # par env : deque d'actions brutes (a,)

    def _span(self):
        return self.cfg.action_block * (self.context - 1)

    def set_initial_history(self, frames, actions):
        """frames[i] : liste des span frames brutes AVANT le départ (la plus
        ancienne d'abord) ; actions[i] : les span actions brutes correspondantes."""
        span = self._span()
        self._frames = [deque(f[-span:] if span else [], maxlen=span + 1) for f in frames]
        self._acts = [deque(a[-span:] if span else [], maxlen=max(span, 1)) for a in actions]

    def get_action(self, info_dict, **kwargs):
        n = self.env.num_envs
        span, blk = self._span(), self.cfg.action_block
        if self._frames is None:
            self.set_initial_history([[] for _ in range(n)], [[] for _ in range(n)])
        px = np.asarray(info_dict['pixels'])
        cur = px[:, -1] if px.ndim == 5 else px          # (n, h, w, c)
        for i in range(n):
            self._frames[i].append(cur[i])
        if self.context > 1:
            frames = np.stack([np.stack([fr[-1 - span + k * blk] for k in range(self.context)])
                               for fr in self._frames])  # (n, H, h, w, c)
            info_dict = dict(info_dict)
            info_dict['pixels'] = frames
            acts = np.stack([np.asarray(list(a)[-span:], dtype=np.float32) for a in self._acts])
            if 'action' in self.process:                 # même normalisation que le plan
                shp = acts.shape
                acts = self.process['action'].transform(acts.reshape(-1, shp[-1])).reshape(shp)
            info_dict['action_history'] = acts.reshape(n, self.context - 1, -1).astype(np.float32)
        action = super().get_action(info_dict, **kwargs)
        for i in range(n):
            self._acts[i].append(np.asarray(action[i], dtype=np.float32).reshape(-1))
        return action


class OraclePolicy(swm.policy.BasePolicy):
    """Plafond du protocole : CEM sur le vrai simulateur. Au premier appel, chaque
    env est copié et la séquence d'actions qui minimise l'erreur de position au pas
    CHECK_STEP est cherchée en simulation ; puis elle est exécutée."""

    def __init__(self, check_step=25, num_samples=300, n_steps=30, topk=30, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.type = 'oracle'
        self.check_step, self.num_samples = check_step, num_samples
        self.n_steps, self.topk = n_steps, topk
        self.rng = np.random.default_rng(seed)
        self._plans = None

    def set_env(self, env):
        self.env = env

    def _plan(self, base):
        low = float(base.action_space.low[0])
        high = float(base.action_space.high[0])
        T = self.check_step
        mu, sd = np.zeros(T), np.full(T, (high - low) / 2)
        for _ in range(self.n_steps):
            seqs = np.clip(mu + sd * self.rng.standard_normal((self.num_samples, T)), low, high)
            costs = []
            for seq in seqs:
                e = copy.copy(base)
                e.state = np.array(base.state, dtype=np.float64).copy()
                for a in seq:
                    type(base).__mro__[2].step(e, np.array([a], dtype=np.float32))
                costs.append(e._pos_error())
            elite = seqs[np.argsort(costs)[:self.topk]]
            mu, sd = elite.mean(0), elite.std(0) + 1e-3
        return list(mu)

    def get_action(self, info_dict, **kwargs):
        envs = self.env.envs
        if self._plans is None:
            self._plans = [self._plan(e.unwrapped) for e in envs]
        out = []
        for p in self._plans:
            out.append([p.pop(0)] if p else [0.0])
        return np.asarray(out, dtype=np.float32).reshape(self.env.action_space.shape)
