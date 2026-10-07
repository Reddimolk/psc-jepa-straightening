"""Idee 9 (PSC) : predicteur en espace des phases + loss L_scal.

Voir "Idee 9 — Mathematiques : espace des phases et L_scal" (28/09/2026).

Architecture (section 3 du doc) :
    u_t     := z_t - z_{t-1}                      vitesse, calculee, sans parametre
    u_hat   := h_theta(...)                       seule dynamique apprise
    z_hat   := z_t + u_hat                        integration codee en dur

Deux variantes de h_theta (option model.phase_history) :
  - Markov strict (defaut, campagnes i9/r9) : h_theta(z_t, u_t, a_t). Le token
    [z_t ; u_t] passe SEUL (sequence de longueur 1) dans le Predictor LeWM.
  - historique (phase_history=true, 07/10/2026) : les tokens [z_i ; u_i] des 3 pas
    de la fenetre passent ensemble dans le Predictor (attention causale, action a_i
    en AdaLN sur chaque token), comme LeWM : meme information (3 frames, 3 actions)
    que la baseline, seule l'integration explicite differe.

L_scal (section 5) : J_u = d h_theta / d u_t, derivee par rapport a la vitesse du
DERNIER pas de la fenetre (le reste de la fenetre fixe). Pour une sonde w (||w|| = 1) :
    J_u w     ~ [h(.., u_t + eps w, ..) - h(.., u_t - eps w, ..)] / (2 eps)
    gamma_hat = <w, J_u w>
    L_scal    = ||J_u w - gamma_hat w||^2                 (version PDF, / ||w||^2 = 1)
    L_scal^inv= ||J_u w - gamma_hat w||^2 / ||J_u w||^2   (normalisee = sin^2(w, J_u w))
Sonde (loss.scal.probe) : 'random' (w uniforme sur la sphere, comme le PDF) ou 'u'
(w = u_t / ||u_t||, la direction reellement visitee : avec des sondes aleatoires en
dimension 192, gamma_hat ~ tr(J_u)/d et ||J_u w|| sont petits des que J_u est de
rang faible, meme si h_theta utilise la vitesse).
"""

import contextlib
import json
import math
import os
import time

import torch
from einops import rearrange
from torch import nn

from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import Attention


class PhaseSpaceLeWM(LeWM):
    """LeWM dont le predicteur agit en espace des phases (z_t, u_t)."""

    def __init__(self, encoder, predictor, action_encoder, projector=None,
                 pred_proj=None, phase_history=False, **kwargs):
        super().__init__(encoder, predictor, action_encoder, projector,
                         pred_proj, **kwargs)
        d = predictor.input_dim
        self.phase_proj = nn.Linear(2 * d, d)
        self.phase_history = phase_history

    @staticmethod
    def velocity(emb):
        """u_t = z_t - z_{t-1} ; u_0 = 0 par convention (vitesse inconnue)."""
        return torch.cat(
            [torch.zeros_like(emb[:, :1]), emb[:, 1:] - emb[:, :-1]], dim=1
        )

    def h(self, z, u, act_emb):
        """h_theta sur une fenetre : z, u (B, T, D), act_emb (B, T, A) -> u_hat
        (B, T, D), u_hat[:, i] = vitesse predite pour le pas i -> i+1."""
        b = z.size(0)
        tok = self.phase_proj(torch.cat([z, u], dim=-1))
        if self.phase_history:
            out = self.predictor(tok, act_emb)            # attention causale
        else:
            out = self.predictor(rearrange(tok, 'b t d -> (b t) 1 d'),
                                 rearrange(act_emb, 'b t d -> (b t) 1 d'))
            out = rearrange(out, '(b t) 1 d -> b t d', b=b)
        out = self.pred_proj(rearrange(out, 'b t d -> (b t) d'))
        return rearrange(out, '(b t) d -> b t d', b=b)

    def predict(self, emb, act_emb):
        """Meme signature que LeWM.predict (donc rollout/planning inchanges) :
        emb (B, T, D), act_emb (B, T, A) -> z_hat_{t+1} pour chaque t, (B, T, D).
        La position 0 a u_0 = 0 (cf. wm.phase_include_first)."""
        return emb + self.h(emb, self.velocity(emb), act_emb)


@contextlib.contextmanager
def no_dropout(module):
    """Coupe le dropout (nn.Dropout + dropout d'attention) sans toucher aux
    BatchNorm, qui restent en mode train : les deux moities +eps/-eps doivent
    voir exactement le meme reseau pour que la difference finie ait un sens."""
    mods = [m for m in module.modules() if isinstance(m, (nn.Dropout, Attention))]
    states = [m.training for m in mods]
    for m in mods:
        m.training = False
    try:
        yield
    finally:
        for m, st in zip(mods, states):
            m.training = st


def scal_terms(model, z, u, act_emb, eps=1e-2, probe='random'):
    """J_u w au dernier pas de la fenetre, par difference finie centree, en
    fp32, entrees detachees (L_scal ne contraint que h_theta, pas l'encodeur).

    z, u, act_emb : (B, T, D). probe : 'random' ou 'u' (w = u_t / ||u_t||).
    Retourne un dict de tenseurs (B,) :
      pdf   = ||J w - gamma w||^2            (||w|| = 1)
      inv   = pdf / ||J w||^2                (= sin^2 de l'angle (w, J w))
      gamma = <w, J w>                       (coefficient scalaire local)
      jw    = ||J w||                        (gain dans la direction w)
    """
    z, u, a = (x.detach().float() for x in (z, u, act_emb))
    n = z.size(0)
    if probe == 'u':
        w = u[:, -1].clone()
    elif probe == 'random':
        w = torch.randn_like(u[:, -1])
    else:
        raise ValueError(f'sonde inconnue : {probe!r}')
    w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    up, um = u.clone(), u.clone()
    up[:, -1] += eps * w
    um[:, -1] -= eps * w
    # Une seule passe sur le batch concatene [u + eps w ; u - eps w] : les
    # statistiques de BatchNorm (pred_proj) sont identiques pour les deux
    # moities, et la moyenne du batch ne bouge pas (perturbations symetriques).
    with torch.autocast(device_type=z.device.type, enabled=False), no_dropout(model):
        out = model.h(torch.cat([z, z]), torch.cat([up, um]),
                      torch.cat([a, a]))[:, -1].float()
    jw = (out[:n] - out[n:]) / (2 * eps)
    gamma = (w * jw).sum(-1)
    res = (jw - gamma.unsqueeze(-1) * w).pow(2).sum(-1)
    jw2 = jw.pow(2).sum(-1)
    return {'pdf': res, 'inv': res / jw2.clamp_min(1e-12),
            'gamma': gamma, 'jw': jw2.sqrt()}


class MetricsLog:
    """Moyennes des metriques ecrites dans $PSC_RUN_DIR/metrics.jsonl (le logger
    Lightning est desactive quand wandb l'est : sans ca, rien n'est conserve).
    Une ligne JSON par bloc. Train : toutes les `every` iterations. Val : une ligne
    par passe de validation complete (ecrite au premier appel train qui suit, ou a
    la fin)."""

    def __init__(self, path, every=200):
        self.path, self.every = path, every
        self.acc, self.cnt, self.n, self.stage, self.last = {}, {}, 0, None, (0, 0)
        self.t0 = time.time()

    def _write(self):
        if not self.n:
            return
        row = {'time_s': round(time.time() - self.t0, 1), 'stage': self.stage,
               'epoch': self.last[0], 'step': self.last[1], 'n': self.n}
        row.update({k: self.acc[k] / self.cnt[k] for k in sorted(self.acc)})
        os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
        with open(self.path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        self.acc, self.cnt, self.n = {}, {}, 0

    def update(self, stage, epoch, step, values):
        if stage != self.stage:
            self._write()
            self.stage = stage
        for k, v in values.items():
            v = float(v)
            if math.isfinite(v):
                self.acc[k] = self.acc.get(k, 0.0) + v
                self.cnt[k] = self.cnt.get(k, 0) + 1
        self.n += 1
        self.last = (epoch, step)
        if stage in ('fit', 'train') and self.n >= self.every:
            self._write()

    def close(self):
        self._write()
