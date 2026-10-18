"""Tests du latent spatial (scripts/train/spatial.py), sur CPU.

1. Formes : encode -> (B, T, P * d), predict, rollout de planification (avec historique).
2. Masque causal par image : la sortie de l'image t ne depend pas des images > t, et
   un patch voit les AUTRES patchs (le predicteur ne traite pas les patchs separement).
3. Idee 9 spatiale : z_hat = z + h(z, u, a), u par patch.
4. L_scal a travers la tete d'agregation : ~0 si J_u = gamma I (tetes mlp et mean),
   > 0 sinon ; gradient vers h et vers la tete.
5. lejepa_forward complet (SIGReg par patch, L_scal, L_curv sur la tete, multi-blocs).

Lancer avec :  python tests/test_spatial.py
"""

import os
import sys

import torch
from torch import nn

ROOT = os.path.join(os.path.dirname(__file__), '..', 'scripts')
sys.path.insert(0, os.path.join(ROOT, 'train'))
sys.path.insert(0, os.path.join(ROOT, 'plan'))
from phase_space import scal_terms  # noqa: E402
from spatial import AggHead, SpatialLeWM, SpatialPredictor  # noqa: E402

torch.manual_seed(0)
E, d, A, G = 24, 6, 4, 4          # dim ViT, dim patch, dim action, grille 4 x 4 -> pool 2 -> 4 patchs
P = 4


def tiny(phase=False, kind='mlp'):
    from stable_worldmodel.wm.lewm.module import MLP, Embedder

    class Enc(nn.Module):  # remplace le ViT : (N, 3, 8, 8) -> CLS + 16 tokens de patch
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3 * 2 * 2, E)

        def forward(self, x, interpolate_pos_encoding=True):
            t = x.unfold(2, 2, 2).unfold(3, 2, 2)                  # (N, 3, 4, 4, 2, 2)
            t = t.permute(0, 2, 3, 1, 4, 5).reshape(x.size(0), G * G, -1)
            h = self.lin(t)
            h = torch.cat([h.mean(1, keepdim=True), h], 1)
            return type('O', (), {'last_hidden_state': h})()

    pred = SpatialPredictor(num_frames=3, num_patches=P, depth=2, heads=2, mlp_dim=32,
                            input_dim=d, hidden_dim=E, output_dim=E, dim_head=8, dropout=0.1)
    bn = lambda n: nn.BatchNorm1d(n)
    return SpatialLeWM(Enc(), pred, Embedder(input_dim=A, emb_dim=E),
                       MLP(E, 32, d, norm_fn=bn), MLP(E, 32, d, norm_fn=bn),
                       agg=AggHead(P, d, kind=kind, hidden_dim=32, out_dim=8), pool=2, phase=phase)


def randomize(m):
    # AdaLN-zero : a l'init les blocs sont l'identite -> poids aleatoires pour tester la structure
    with torch.no_grad():
        for prm in m.predictor.parameters():
            prm.add_(0.3 * torch.randn_like(prm))


def test_shapes_and_rollout():
    import psc_plan
    for phase in (False, True):
        m = tiny(phase).eval()
        info = m.encode({'pixels': torch.randn(2, 3, 3, 8, 8), 'action': torch.randn(2, 3, A)})
        assert info['emb'].shape == (2, 3, P * d)
        assert m.predict(info['emb'], info['act_emb']).shape == (2, 3, P * d)
        r = m.rollout({'pixels': torch.randn(2, 4, 1, 3, 8, 8)}, torch.randn(2, 4, 1 + 5, A))
        assert r['predicted_emb'].shape == (2, 4, 7, P * d)   # H + (T - H) + 1
        psc_plan.install_history_rollout(m)
        r = m.rollout({'pixels': torch.randn(2, 4, 2, 3, 8, 8),
                       'action_history': torch.randn(2, 4, 1, A)}, torch.randn(2, 4, 5, A))
        assert torch.isfinite(r['predicted_emb']).all() and r['predicted_emb'].shape[2] == 7
    print('formes, rollout 0.1.1 et rollout avec historique : OK')


def test_frame_causal_and_spatial_mixing():
    for phase in (False, True):
        m = tiny(phase).eval()
        randomize(m)
        emb, act = torch.randn(3, 3, P * d), torch.randn(3, 3, E)
        out = m.predict(emb, act)
        e2 = emb.clone()
        e2[:, 2] += 1.0                               # image 2 modifiee
        out2 = m.predict(e2, act)
        assert torch.allclose(out[:, :2], out2[:, :2], atol=1e-5), 'fuite du futur'
        e3 = emb.clone()
        e3[:, 0, :d] += 1.0                           # patch 0 de l'image 0 seulement
        out3 = m.predict(e3, act)
        o, o3 = (m.tokens(x) for x in (out, out3))
        assert not torch.allclose(o[:, 1, 3], o3[:, 1, 3]), 'patch 3 (image 1) ne voit pas le patch 0 (image 0)'
        if phase:
            assert torch.allclose(out, emb + m.h(emb, m.velocity(emb), act), atol=1e-5)
    print('masque causal par image, melange entre patchs, integration Idee 9 : OK')


class FakeSpatial(nn.Module):
    """h = u @ J^T + g(z, a) sur des vecteurs aplatis, J connu."""

    def __init__(self, J):
        super().__init__()
        self.J = nn.Parameter(J.clone())
        self.g = nn.Linear(P * d + A, P * d)

    def h(self, z, u, a):
        return u @ self.J.T + torch.tanh(self.g(torch.cat([z, a], -1)))


def test_scal_through_agg():
    n, D = 256, P * d
    z, u, a = torch.randn(n, 3, D), torch.randn(n, 3, D), torch.randn(n, 3, A)
    for kind in ('mlp', 'mean'):
        agg = AggHead(P, d, kind=kind, hidden_dim=32, out_dim=8)
        for probe in ('u', 'random'):
            t = scal_terms(FakeSpatial(0.8 * torch.eye(D)), z, u, a, probe=probe, agg=agg)
            assert t['inv'].mean() < 1e-4, (kind, probe, t['inv'].mean())
            assert torch.allclose(t['gamma'], torch.full((n,), 0.8), atol=1e-2)
        J = torch.diag(torch.linspace(-1, 1, D)) + 0.3 * torch.randn(D, D)
        m = FakeSpatial(J)
        t = scal_terms(m, z, u, a, probe='u', agg=agg)
        assert t['inv'].mean() > 0.05
        t['inv'].mean().backward()
        assert m.J.grad.abs().sum() > 0
        if kind == 'mlp':
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in agg.parameters())
        print(f'L_scal a travers la tete {kind} : J = 0.8 I -> ~0, J quelconque -> {t["inv"].mean():.3f}')


def test_lejepa_forward():
    from omegaconf import OmegaConf
    from stable_worldmodel.wm.loss import SIGReg
    os.environ.pop('PSC_RUN_DIR', None)
    import lewm

    cfg = OmegaConf.create({
        'wm': {'history_size': 3, 'num_preds': 1, 'phase_include_first': True},
        'loss': {'sigreg': {'weight': 0.09}, 'jac': {'weight': 0.0, 'eps': 1e-2},
                 'scal': {'weight': 0.1, 'eps': 1e-2, 'norm': 'inv', 'probe': 'u'},
                 'multi': {'k': 5, 'weight': 1.0}, 'curv': {'weight': 0.1}}})
    for phase, ckpt in ((False, False), (True, False), (True, True)):
        m = tiny(phase)
        m.predictor.grad_ckpt = ckpt

        class Host(nn.Module):
            current_epoch, global_step = 0, 0

            def __init__(self):
                super().__init__()
                self.model = m
                self.sigreg = SIGReg(knots=17, num_proj=64)

            def log_dict(self, *a, **k):
                pass

        h = Host().train()
        batch = {'pixels': torch.randn(8, 8, 3, 8, 8), 'action': torch.randn(8, 8, A)}
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = lewm.lejepa_forward(h, batch, 'fit', cfg)
        out['loss'].backward()
        for k in ('pred_loss', 'sigreg_loss', 'multi_loss', 'curv_loss', 'loss'):
            assert torch.isfinite(out[k]), k
        assert m.agg.net.net[0].weight.grad is not None, 'L_curv ne passe pas par la tete'
        assert m.encoder.lin.weight.grad is not None
        if phase:
            assert torch.isfinite(out['scal_loss']) and m.phase_proj.weight.grad is not None
        print(f'lejepa_forward (phase={phase}, ckpt={ckpt}) : ' + ' '.join(
            f'{k}={out[k].item():.4f}' for k in out if 'loss' in k))


if __name__ == '__main__':
    test_shapes_and_rollout()
    test_frame_causal_and_spatial_mixing()
    test_scal_through_agg()
    test_lejepa_forward()
    print('\nTous les tests sont passes.')
