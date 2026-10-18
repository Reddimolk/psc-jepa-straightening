"""Latent spatial (par patchs), comme Temporal Straightening (Wang et al. 2026), pour
LeWM et l'Idee 9 (18/10/2026).

LeWM ne garde que le token CLS du ViT : une image = un vecteur. Ici on garde les
tokens de patch (16 x 16 pour 224 px / patchs de 14), regroupes `pool` x `pool`
(8 x 8 = 64 patchs par defaut) et projetes token par token en dimension d :
une image = P vecteurs z^i (i = 1..P).

Stockage : z est APLATI en (B, T, P * d). Tout ce qui traite z comme un vecteur
reste valable tel quel : rollout / planification (cout = somme des L2 des patchs,
comme le papier), loss multi-blocs, erreur de rollout, paysage de cout. Seuls le
predicteur, SIGReg et la tete d'agregation remettent la forme (P, d).

Predicteur : un seul transformeur sur les T * P tokens (image t, patch i), avec
position spatiale + position temporelle, action de l'image t en AdaLN sur chacun
de ses patchs, et masque causal PAR IMAGE (un token de l'image t voit tous les
patchs des images <= t). Sortie : un token par (t, i) = prediction de z^i_{t+1}.

Idee 9 (phase=True) : u^i_t = z^i_t - z^i_{t-1} par patch, token [z^i_t ; u^i_t],
z_hat^i_{t+1} = z^i_t + u_hat^i_{t+1}.

Tete d'agregation A (papier, Fig. 13) : vecteurs vitesse (P * d) -> vecteur global
(128). L_curv = 1 - cos(A(v_t), A(v_{t+1})) ; L_scal est mesuree a travers A
(phase_space.scal_terms, argument agg).
"""

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from stable_worldmodel.wm.lewm import LeWM
from stable_worldmodel.wm.lewm.module import (
    MLP, Attention, ConditionalBlock, Transformer)


class FrameCausalAttention(Attention):
    """Attention avec masque explicite (pose par SpatialPredictor avant la passe)
    au lieu du masque causal token par token de LeWM."""

    mask = None  # (L, L) bool, True = autorise

    def forward(self, x, causal=True):
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        q, k, v = (rearrange(t, 'b t (h d) -> b h t d', h=self.heads)
                   for t in self.to_qkv(x).chunk(3, dim=-1))
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask, dropout_p=drop)
        return self.to_out(rearrange(out, 'b h t d -> b t (h d)'))


class SpatialBlock(ConditionalBlock):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__(dim, heads, dim_head, mlp_dim, dropout)
        self.attn = FrameCausalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)


class SpatialPredictor(nn.Module):
    """x (B, T, P, input_dim), c (B, T, hidden_dim) -> (B, T, P, hidden_dim)."""

    def __init__(self, *, num_frames, num_patches, depth, heads, mlp_dim, input_dim,
                 hidden_dim, output_dim=None, dim_head=64, dropout=0.0, emb_dropout=0.0):
        super().__init__()
        self.num_frames, self.num_patches = num_frames, num_patches
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.time_pos = nn.Parameter(torch.randn(1, num_frames, 1, hidden_dim))
        self.space_pos = nn.Parameter(torch.randn(1, 1, num_patches, hidden_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(hidden_dim, hidden_dim, hidden_dim, depth, heads,
                                       dim_head, mlp_dim, dropout, block_class=SpatialBlock)

    def forward(self, x, c):
        b, t, p, _ = x.shape
        x = self.in_proj(x) + self.time_pos[:, :t] + self.space_pos
        x = self.dropout(rearrange(x, 'b t p d -> b (t p) d'))
        c = c.unsqueeze(2).expand(-1, -1, p, -1).reshape(b, t * p, -1)
        frame = torch.arange(t, device=x.device).repeat_interleave(p)
        mask = frame[None, :] <= frame[:, None]          # image du token vu <= la sienne
        for blk in self.transformer.layers:
            blk.attn.mask = mask
        out = self.transformer(x, c)
        return rearrange(out, 'b (t p) d -> b t p d', t=t)


class AggHead(nn.Module):
    """Tete d'agregation : (..., P * d) -> (..., out_dim).
    kind='mlp' : MLP appris (papier : sortie 128) ; kind='mean' : moyenne des
    patchs, sans parametre (variante « mean » du papier, Fig. 14)."""

    def __init__(self, num_patches, patch_dim, kind='mlp', hidden_dim=512, out_dim=128):
        super().__init__()
        self.kind, self.num_patches = kind, num_patches
        if kind == 'mlp':
            self.net = MLP(num_patches * patch_dim, hidden_dim, out_dim)   # LayerNorm
        elif kind != 'mean':
            raise ValueError(f'tete inconnue : {kind!r}')

    def forward(self, v):
        if self.kind == 'mean':
            return rearrange(v, '... (p d) -> ... p d', p=self.num_patches).mean(-2)
        lead = v.shape[:-1]
        return self.net(v.reshape(-1, v.size(-1))).reshape(*lead, -1)


class SpatialLeWM(LeWM):
    """LeWM (phase=False) ou Idee 9 (phase=True) sur un latent de P patchs."""

    def __init__(self, encoder, predictor, action_encoder, projector=None, pred_proj=None,
                 agg=None, pool=2, phase=False, **kwargs):
        super().__init__(encoder, predictor, action_encoder, projector, pred_proj, **kwargs)
        self.pool, self.phase, self.agg = pool, phase, agg
        self.num_patches = predictor.num_patches
        d = predictor.input_dim
        self.phase_proj = nn.Linear(2 * d, d) if phase else None

    def encode(self, info):
        pixels = info['pixels'].to(next(self.encoder.parameters()).dtype)
        b = pixels.size(0)
        pixels = rearrange(pixels, 'b t ... -> (b t) ...')
        x = self.encoder(pixels, interpolate_pos_encoding=True).last_hidden_state[:, 1:]
        g = math.isqrt(x.size(1))                           # grille g x g (sans le CLS)
        x = rearrange(x, 'n (h w) c -> n c h w', h=g)
        if self.pool > 1:
            x = F.avg_pool2d(x, self.pool)
        p = x.size(-1) * x.size(-2)
        assert p == self.num_patches, f'{p} patchs, le predicteur en attend {self.num_patches}'
        emb = self.projector(rearrange(x, 'n c h w -> (n h w) c'))
        info['emb'] = rearrange(emb, '(b t p) d -> b t (p d)', b=b, p=p)
        if 'action' in info:
            info['act_emb'] = self.action_encoder(info['action'])
        return info

    def tokens(self, emb):
        """(B, T, P * d) -> (B, T, P, d)"""
        return rearrange(emb, 'b t (p d) -> b t p d', p=self.num_patches)

    def _run(self, x, act_emb):
        out = self.predictor(x, act_emb)                    # (B, T, P, hidden)
        b, t, p = out.shape[:3]
        out = self.pred_proj(rearrange(out, 'b t p c -> (b t p) c'))
        return rearrange(out, '(b t p) d -> b t (p d)', b=b, t=t)

    @staticmethod
    def velocity(emb):
        """u_t = z_t - z_{t-1} (patch par patch, aplati) ; u_0 = 0."""
        return torch.cat([torch.zeros_like(emb[:, :1]), emb[:, 1:] - emb[:, :-1]], dim=1)

    def h(self, z, u, act_emb):
        """Idee 9 : u_hat (B, T, P * d) a partir de z, u (B, T, P * d)."""
        x = self.phase_proj(torch.cat([self.tokens(z), self.tokens(u)], dim=-1))
        return self._run(x, act_emb)

    def predict(self, emb, act_emb):
        if self.phase:
            return emb + self.h(emb, self.velocity(emb), act_emb)
        return self._run(self.tokens(emb), act_emb)
