"""
Tests de l'Idee 9 (espace des phases) et de L_scal, sur CPU.

1. L_scal sur des predicteurs dont on connait J_u exactement :
   - h = gamma * u + g(z, a)   -> J_u = gamma I     -> L_scal (pdf et inv) ~ 0
   - h = M u (M anisotrope)    -> L_scal > 0, egale a la valeur analytique
   - h = 0 * u (Aristote)      -> L_scal pdf = 0 (solution triviale), inv mal defini
2. Le gradient de L_scal remonte jusqu'aux parametres du predicteur, pas a l'encodeur.
3. PhaseSpaceLeWM : z_hat = z + h(z, u, a), Markov strict (la sortie en t ne depend
   pas de z_{t-2}), compatible avec LeWM.rollout.
4. lejepa_forward complet (pred + sigreg + scal) sur un mini modele et un faux batch.

Lancer avec :  python tests/test_scal_loss.py
"""

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'train'))
from phase_space import PhaseSpaceLeWM, scal_terms  # noqa: E402

torch.manual_seed(0)
D, A, N = 16, 4, 512


class FakeModel(nn.Module):
    """Expose phase_step(z, u, a) comme PhaseSpaceLeWM, avec J_u connu."""

    def __init__(self, J):
        super().__init__()
        self.J = nn.Parameter(J.clone())
        self.g = nn.Linear(D + A, D)
        self.drop = nn.Dropout(0.5)  # doit etre coupe par no_dropout

    def phase_step(self, z, u, a):
        return u @ self.J.T + self.drop(torch.tanh(self.g(torch.cat([z, a], -1))))


def expected_pdf(J, n=200_000):
    """E_w ||J w - <w, J w> w||^2 pour w uniforme sur la sphere, par Monte-Carlo."""
    w = torch.randn(n, D)
    w = w / w.norm(dim=-1, keepdim=True)
    jw = w @ J.T
    g = (w * jw).sum(-1, keepdim=True)
    return (jw - g * w).pow(2).sum(-1).mean().item()


def data():
    return torch.randn(N, D), torch.randn(N, D), torch.randn(N, A)


def test_scalar_jacobian():
    m = FakeModel(0.7 * torch.eye(D)).train()
    t = scal_terms(m, *data())
    assert t['pdf'].mean() < 1e-6, t['pdf'].mean()
    assert t['inv'].mean() < 1e-6, t['inv'].mean()
    assert torch.allclose(t['gamma'], torch.full((N,), 0.7), atol=1e-4)
    print(f"J = 0.7 I          : pdf={t['pdf'].mean():.2e} inv={t['inv'].mean():.2e} "
          f"gamma={t['gamma'].mean():.4f}")


def test_anisotropic_jacobian():
    J = torch.diag(torch.linspace(0.1, 1.0, D)) + 0.1 * torch.randn(D, D)
    m = FakeModel(J).train()
    t = scal_terms(m, *data())
    ref = expected_pdf(J)
    assert abs(t['pdf'].mean().item() - ref) / ref < 0.1, (t['pdf'].mean(), ref)
    assert 0 < t['inv'].mean() < 1
    print(f"J anisotrope       : pdf={t['pdf'].mean():.4f} (analytique {ref:.4f}) "
          f"inv={t['inv'].mean():.4f}")


def test_aristote_trivial():
    m = FakeModel(torch.zeros(D, D)).train()
    t = scal_terms(m, *data())
    assert t['pdf'].mean() < 1e-8
    print(f"J = 0 (Aristote)   : pdf={t['pdf'].mean():.2e}  <- solution triviale de la version PDF")


def test_gradient_only_to_predictor():
    J = torch.randn(D, D) / D**0.5
    m = FakeModel(J).train()
    z, u, a = (x.requires_grad_(True) for x in data())
    for norm in ('pdf', 'inv'):
        m.zero_grad()
        scal_terms(m, z, u, a)[norm].mean().backward()
        assert m.J.grad is not None and m.J.grad.abs().sum() > 0
        assert z.grad is None and u.grad is None and a.grad is None
    # la descente de gradient sur L_scal rend bien J scalaire
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(500):
        opt.zero_grad()
        scal_terms(m, *data())['inv'].mean().backward()
        opt.step()
    off = (m.J - torch.diag(m.J).mean() * torch.eye(D)).norm() / m.J.norm()
    print(f"apres 500 pas d'Adam sur L_scal^inv : ||J - gamma I|| / ||J|| = {off:.3f}")
    assert off < 0.1


def tiny_phase_model():
    from stable_worldmodel.wm.lewm.module import MLP, Embedder, Predictor

    class Enc(nn.Module):  # remplace le ViT : pixels (N, 3, 8, 8) -> cls token
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(3 * 8 * 8, D)

        def forward(self, x, interpolate_pos_encoding=True):
            h = self.lin(x.flatten(1)).unsqueeze(1)
            return type('O', (), {'last_hidden_state': h})()

    pred = Predictor(num_frames=3, depth=2, heads=2, mlp_dim=32, input_dim=D,
                     hidden_dim=D, output_dim=D, dim_head=8, dropout=0.1)
    bn = lambda d: nn.BatchNorm1d(d)
    return PhaseSpaceLeWM(Enc(), pred, Embedder(input_dim=A, emb_dim=D),
                          MLP(D, 32, D, norm_fn=bn), MLP(D, 32, D, norm_fn=bn))


def test_phase_model_markov():
    m = tiny_phase_model().eval()
    emb, act = torch.randn(5, 3, D), torch.randn(5, 3, D)
    out = m.predict(emb, act)
    assert out.shape == emb.shape
    emb2 = emb.clone()
    emb2[:, 0] += 1.0  # ne doit changer que les positions 0 et 1 (u_1 = z_1 - z_0)
    out2 = m.predict(emb2, act)
    assert torch.allclose(out[:, 2], out2[:, 2], atol=1e-6), 'h_theta pas Markov'
    assert not torch.allclose(out[:, 1], out2[:, 1])
    # z_hat = z + u_hat : l'integration est bien codee en dur
    u = m.velocity(emb)
    u_hat = m.phase_step(emb[:, 2], u[:, 2], act[:, 2])
    assert torch.allclose(out[:, 2], emb[:, 2] + u_hat, atol=1e-6)
    # rollout de planification (code LeWM inchange)
    # API rollout de stable-worldmodel 0.1.1 : action_sequence couvre les H
    # pas de contexte + l'horizon
    info = {'pixels': torch.randn(2, 4, 3, 3, 8, 8)}
    r = m.rollout(info, torch.randn(2, 4, 3 + 5, A))
    assert r['predicted_emb'].shape[-1] == D and torch.isfinite(r['predicted_emb']).all()
    print('PhaseSpaceLeWM     : Markov strict, integration explicite, rollout OK')


def test_lejepa_forward():
    from omegaconf import OmegaConf
    from stable_worldmodel.wm.loss import SIGReg
    os.environ.pop('PSC_RUN_DIR', None)
    import lewm

    cfg = OmegaConf.create({
        'wm': {'history_size': 3, 'num_preds': 1},
        'loss': {'sigreg': {'weight': 0.09}, 'jac': {'weight': 0.1, 'eps': 1e-2},
                 'scal': {'weight': 1.0, 'eps': 1e-2, 'norm': 'inv'}}})

    class Host(nn.Module):  # remplace spt.Module : juste ce que lejepa_forward utilise
        current_epoch, global_step = 0, 0

        def __init__(self):
            super().__init__()
            self.model = tiny_phase_model()
            self.sigreg = SIGReg(knots=17, num_proj=64)

        def log_dict(self, *a, **k):
            pass

    h = Host().train()
    batch = {'pixels': torch.randn(8, 4, 3, 8, 8), 'action': torch.randn(8, 4, A)}
    batch['action'][0, 0] = float('nan')
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out = lewm.lejepa_forward(h, batch, 'fit', cfg)
    out['loss'].backward()
    for k in ('pred_loss', 'sigreg_loss', 'jac_loss', 'scal_loss', 'loss'):
        assert torch.isfinite(out[k]), k
    assert h.model.phase_proj.weight.grad is not None
    print('lejepa_forward     : ' + ' '.join(
        f'{k}={out[k].item():.4f}' for k in ('pred_loss', 'scal_loss', 'jac_loss', 'loss')))


if __name__ == '__main__':
    test_scalar_jacobian()
    test_anisotropic_jacobian()
    test_aristote_trivial()
    test_gradient_only_to_predictor()
    test_phase_model_markov()
    test_lejepa_forward()
    print('\nTous les tests sont passes.')
