"""Idee 9 (PSC) : predicteur en espace des phases + loss L_scal.

Voir "Idee 9 — Mathematiques : espace des phases et L_scal" (28/09/2026).

Architecture (section 3 du doc) :
    u_t     := z_t - z_{t-1}                      vitesse, calculee, sans parametre
    u_hat   := h_theta(z_t, u_t, a_t)             seule dynamique apprise
    z_hat   := z_t + u_hat                        integration codee en dur

h_theta est Markov strict : il ne voit que (z_t, u_t, a_t), pas l'historique.
Concretement, le token [z_t ; u_t] (dim 2D) est projete en dim D par `phase_proj`,
puis passe SEUL (sequence de longueur 1) dans le Predictor LeWM (conditionnement
AdaLN par l'action inchange), puis dans pred_proj -> u_hat.

L_scal (section 5) : pour une sonde w (une par echantillon, ||w|| = 1),
    J_u w     ~ [h(z, u + eps w, a) - h(z, u - eps w, a)] / (2 eps)
    gamma_hat = <w, J_u w>
    L_scal    = ||J_u w - gamma_hat w||^2                 (version PDF, / ||w||^2 = 1)
    L_scal^inv= ||J_u w - gamma_hat w||^2 / ||J_u w||^2   (invariante d'echelle,
                = sin^2(w, J_u w) ; insensible a la solution triviale J_u -> 0)
"""

import contextlib
import csv
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
                 pred_proj=None, **kwargs):
        super().__init__(encoder, predictor, action_encoder, projector,
                         pred_proj, **kwargs)
        d = predictor.input_dim
        self.phase_proj = nn.Linear(2 * d, d)

    @staticmethod
    def velocity(emb):
        """u_t = z_t - z_{t-1} ; u_0 = 0 par convention (vitesse inconnue)."""
        return torch.cat(
            [torch.zeros_like(emb[:, :1]), emb[:, 1:] - emb[:, :-1]], dim=1
        )

    def phase_step(self, z, u, act_emb):
        """h_theta(z, u, a) -> u_hat, sur des tenseurs a plat (N, D)."""
        tok = self.phase_proj(torch.cat([z, u], dim=-1))
        out = self.predictor(tok.unsqueeze(1), act_emb.unsqueeze(1))[:, 0]
        return self.pred_proj(out)

    def predict(self, emb, act_emb):
        """Meme signature que LeWM.predict (donc rollout/planning inchanges) :
        emb (B, T, D), act_emb (B, T, A) -> z_hat_{t+1} pour chaque t, (B, T, D).
        La position 0 a u_0 = 0 : elle est exclue de la loss d'entrainement."""
        b = emb.size(0)
        u = self.velocity(emb)
        flat = lambda x: rearrange(x, 'b t d -> (b t) d')
        u_hat = self.phase_step(flat(emb), flat(u), flat(act_emb))
        return emb + rearrange(u_hat, '(b t) d -> b t d', b=b)


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
        for m, s in zip(mods, states):
            m.training = s


def scal_terms(model, z, u, act_emb, eps=1e-2):
    """J_u w par difference finie centree, en fp32, entrees detachees
    (L_scal ne contraint que le predicteur h_theta, pas l'encodeur).

    z, u, act_emb : (N, D). Retourne un dict de tenseurs (N,) :
      pdf   = ||J w - gamma w||^2            (||w|| = 1)
      inv   = pdf / ||J w||^2                (= sin^2 de l'angle (w, J w))
      gamma = <w, J w>                       (coefficient scalaire local)
      jw    = ||J w||
    """
    z, u, a = (x.detach().float() for x in (z, u, act_emb))
    n = z.size(0)
    w = torch.randn_like(u)
    w = w / w.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    # Une seule passe sur le batch concatene [u + eps w ; u - eps w] : les
    # statistiques de BatchNorm (pred_proj) sont identiques pour les deux
    # moities, et la moyenne du batch ne bouge pas (perturbations symetriques).
    with torch.autocast(device_type=z.device.type, enabled=False), no_dropout(model):
        out = model.phase_step(
            torch.cat([z, z]), torch.cat([u + eps * w, u - eps * w]), torch.cat([a, a])
        ).float()
    jw = (out[:n] - out[n:]) / (2 * eps)
    gamma = (w * jw).sum(-1)
    res = (jw - gamma.unsqueeze(-1) * w).pow(2).sum(-1)
    jw2 = jw.pow(2).sum(-1)
    return {'pdf': res, 'inv': res / jw2.clamp_min(1e-12),
            'gamma': gamma, 'jw': jw2.sqrt()}


class MetricsCSV:
    """Moyennes des metriques ecrites dans $PSC_RUN_DIR/metrics.csv (le logger
    Lightning est desactive quand wandb l'est : sans ca, rien n'est conserve).
    Train : une ligne toutes les `every` iterations. Val : une ligne par passe
    de validation complete (ecrite au premier appel train qui suit, ou a la fin)."""

    def __init__(self, path, every=200):
        self.path, self.every = path, every
        self.acc, self.n, self.stage, self.last = {}, 0, None, (0, 0)
        self.t0 = time.time()
        self._fh = None

    def _write(self):
        if not self.n:
            return
        row = {'time_s': round(time.time() - self.t0, 1), 'stage': self.stage,
               'epoch': self.last[0], 'step': self.last[1], 'n': self.n}
        row.update({k: v / self.n for k, v in sorted(self.acc.items())})
        new = self._fh is None
        if new:
            os.makedirs(os.path.dirname(self.path) or '.', exist_ok=True)
            self._fh = open(self.path, 'a', newline='')
            self._cols = None
        if self._cols is None or set(row) - set(self._cols):
            self._cols = list(row)
            csv.writer(self._fh).writerow(self._cols)
        csv.writer(self._fh).writerow([row.get(c, '') for c in self._cols])
        self._fh.flush()
        self.acc, self.n = {}, 0

    def update(self, stage, epoch, step, values):
        if stage != self.stage:
            self._write()
            self.stage = stage
        for k, v in values.items():
            v = float(v)
            if math.isfinite(v):
                self.acc[k] = self.acc.get(k, 0.0) + v
        self.n += 1
        self.last = (epoch, step)
        if stage in ('fit', 'train') and self.n >= self.every:
            self._write()

    def close(self):
        self._write()
