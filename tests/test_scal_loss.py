"""
Tests de l'Idee 9 (espace des phases) et de L_scal, sur CPU.

1. L_scal sur des predicteurs dont on connait J_u exactement :
   - h = gamma * u + g(z, a)   -> J_u = gamma I     -> L_scal (pdf et inv) ~ 0
   - h = M u (M anisotrope)    -> L_scal > 0, egale a la valeur analytique
   - h = 0 * u (Aristote)      -> L_scal pdf = 0 (solution triviale), inv mal defini
2. Sonde le long de u_t : mesure J_u dans la direction visitee ; un J_u de rang
   faible parait "petit" avec des sondes aleatoires en dimension 192, pas avec u_t.
3. Le gradient de L_scal remonte jusqu'aux parametres du predicteur, pas a l'encodeur.
4. PhaseSpaceLeWM : z_hat = z + h(...), Markov strict (la sortie en t ne depend
   pas de z_{t-2}) ou historique (elle en depend), compatible avec LeWM.rollout.
5. lejepa_forward complet (pred + sigreg + scal, deux sondes, ablations).

Lancer avec :  python tests/test_scal_loss.py
"""

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts', 'train'))
from phase_space import PhaseSpaceLeWM, scal_terms  # noqa: E402

torch.manual_seed(0)
D, A, N, T = 16, 4, 512, 3


class FakeModel(nn.Module):
    """Expose h(z, u, a) sur des fenetres (B, T, .) comme PhaseSpaceLeWM, avec
    J_u = d h[:, -1] / d u[:, -1] connu (= J)."""

    def __init__(self, J, d=D):
        super().__init__()
        self.J = nn.Parameter(J.clone())
        self.g = nn.Linear(d + A, d)
        self.drop = nn.Dropout(0.5)  # doit etre coupe par no_dropout

    def h(self, z, u, a):
        return u @ self.J.T + self.drop(torch.tanh(self.g(torch.cat([z, a], -1))))


def expected_pdf(J, n=200_000):
    """E_w ||J w - <w, J w> w||^2 pour w uniforme sur la sphere, par Monte-Carlo."""
    w = torch.randn(n, J.size(0))
    w = w / w.norm(dim=-1, keepdim=True)
    jw = w @ J.T
    g = (w * jw).sum(-1, keepdim=True)
    return (jw - g * w).pow(2).sum(-1).mean().item()


def data(d=D, n=N):
    return torch.randn(n, T, d), torch.randn(n, T, d), torch.randn(n, T, A)


def test_scalar_jacobian():
    m = FakeModel(0.7 * torch.eye(D)).train()
    for probe in ('random', 'u'):
        t = scal_terms(m, *data(), probe=probe)
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


def test_probe_along_u():
    """J = diag(lambda) ; u_t le long du vecteur propre e_0 : la sonde u voit
    exactement la reponse dans cette direction (gamma = lambda_0, inv = 0)."""
    lam = torch.linspace(0.2, 1.0, D)
    m = FakeModel(torch.diag(lam)).train()
    z, u, a = data()
    u[:, -1] = 0
    u[:, -1, 0] = torch.rand(N) + 0.5  # vitesse visitee le long de e_0
    t = scal_terms(m, z, u, a, probe='u')
    assert torch.allclose(t['gamma'], torch.full((N,), lam[0].item()), atol=1e-4)
    assert t['inv'].mean() < 1e-6
    print(f"sonde le long de u : gamma={t['gamma'].mean():.4f} (= lambda_0 = {lam[0]:.2f}), "
          f"inv={t['inv'].mean():.1e}")


def test_low_rank_random_vs_u():
    """Le piege des sondes aleatoires : J_u de rang 5 et de gain 1 en dimension 192.
    Sondes aleatoires : ||J w|| ~ sqrt(5/192) ~ 0.16 et gamma ~ 5/192 ~ 0.03, comme si
    h ignorait la vitesse. Sonde le long de u_t (dans le sous-espace) : gain 1, gamma 1."""
    d, r = 192, 5
    J = torch.zeros(d, d)
    J[:r, :r] = torch.eye(r)
    m = FakeModel(J, d=d).train()
    z, u, a = data(d=d, n=2048)
    u[:, -1] = 0
    u[:, -1, :r] = torch.randn(2048, r)  # la dynamique ne visite que ces r directions
    tr, tu = scal_terms(m, z, u, a, probe='random'), scal_terms(m, z, u, a, probe='u')
    assert abs(tr['jw'].pow(2).mean().item() - r / d) < 0.01
    assert abs(tr['gamma'].mean().item() - r / d) < 0.01
    assert (tu['jw'] - 1).abs().max() < 1e-3 and (tu['gamma'] - 1).abs().max() < 1e-3
    print(f"rang 5 en dim 192  : sonde aleatoire ||Jw||={tr['jw'].mean():.3f} gamma={tr['gamma'].mean():.3f}"
          f" | sonde u_t ||Jw||={tu['jw'].mean():.3f} gamma={tu['gamma'].mean():.3f}")


def test_gradient_only_to_predictor():
    J = torch.randn(D, D) / D**0.5
    m = FakeModel(J).train()
    z, u, a = (x.requires_grad_(True) for x in data())
    for probe in ('random', 'u'):
        for norm in ('pdf', 'inv'):
            m.zero_grad()
            scal_terms(m, z, u, a, probe=probe)[norm].mean().backward()
            assert m.J.grad is not None and m.J.grad.abs().sum() > 0
            assert z.grad is None and u.grad is None and a.grad is None
    # la descente de gradient sur L_scal (sondes aleatoires) rend bien J scalaire
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    for _ in range(500):
        opt.zero_grad()
        scal_terms(m, *data())['inv'].mean().backward()
        opt.step()
    off = (m.J - torch.diag(m.J).mean() * torch.eye(D)).norm() / m.J.norm()
    print(f"apres 500 pas d'Adam sur L_scal^inv : ||J - gamma I|| / ||J|| = {off:.3f}")
    assert off < 0.1


def tiny_phase_model(history=False):
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
                          MLP(D, 32, D, norm_fn=bn), MLP(D, 32, D, norm_fn=bn),
                          phase_history=history)


def test_phase_model():
    emb, act = torch.randn(5, 3, D), torch.randn(5, 3, D)
    emb2 = emb.clone()
    emb2[:, 0] += 1.0  # change z_0 (donc u_1), mais pas z_1, z_2 ni u_2
    for history in (False, True):
        m = tiny_phase_model(history).eval()
        # AdaLN-zero : a l'init les blocs du transformeur sont l'identite (aucun
        # melange entre pas) -> on randomise les poids pour tester la structure
        with torch.no_grad():
            for prm in m.predictor.parameters():
                prm.add_(0.3 * torch.randn_like(prm))
        out, out2 = m.predict(emb, act), m.predict(emb2, act)
        assert out.shape == emb.shape
        assert not torch.allclose(out[:, 1], out2[:, 1])
        if history:   # le dernier pas voit toute la fenetre
            assert not torch.allclose(out[:, 2], out2[:, 2]), 'historique ignore'
        else:         # Markov strict : le dernier pas ne voit que (z_2, u_2, a_2)
            assert torch.allclose(out[:, 2], out2[:, 2], atol=1e-6), 'h_theta pas Markov'
        # z_hat = z + u_hat : l'integration est bien codee en dur
        u_hat = m.h(emb, m.velocity(emb), act)
        assert torch.allclose(out, emb + u_hat, atol=1e-6)
        # rollout de planification (code LeWM inchange, API 0.1.1)
        info = {'pixels': torch.randn(2, 4, 3, 3, 8, 8)}
        r = m.rollout(info, torch.randn(2, 4, 3 + 5, A))
        assert torch.isfinite(r['predicted_emb']).all()
    print('PhaseSpaceLeWM     : Markov strict et historique, integration explicite, rollout OK')


def test_lejepa_forward():
    from omegaconf import OmegaConf
    from stable_worldmodel.wm.loss import SIGReg
    os.environ.pop('PSC_RUN_DIR', None)
    import lewm

    for history, probe in ((False, 'random'), (True, 'u')):
        cfg = OmegaConf.create({
            'wm': {'history_size': 3, 'num_preds': 1, 'phase_include_first': True},
            'loss': {'sigreg': {'weight': 0.09}, 'jac': {'weight': 0.0, 'eps': 1e-2},
                     'scal': {'weight': 1.0, 'eps': 1e-2, 'norm': 'inv', 'probe': probe}}})

        class Host(nn.Module):  # remplace spt.Module : juste ce que lejepa_forward utilise
            current_epoch, global_step = 0, 0

            def __init__(self):
                super().__init__()
                self.model = tiny_phase_model(history)
                self.sigreg = SIGReg(knots=17, num_proj=64)

            def log_dict(self, *a, **k):
                pass

        h = Host().train()
        batch = {'pixels': torch.randn(8, 4, 3, 8, 8), 'action': torch.randn(8, 4, A)}
        batch['action'][0, 0] = float('nan')
        with torch.autocast('cpu', dtype=torch.bfloat16):
            out = lewm.lejepa_forward(h, batch, 'fit', cfg)
        out['loss'].backward()
        for k in ('pred_loss', 'sigreg_loss', 'scal_loss', 'loss'):
            assert torch.isfinite(out[k]), k
        assert h.model.phase_proj.weight.grad is not None
        print(f'lejepa_forward (historique={history}, sonde={probe}) : ' + ' '.join(
            f'{k}={out[k].item():.4f}' for k in ('pred_loss', 'scal_loss', 'loss')))


if __name__ == '__main__':
    test_scalar_jacobian()
    test_anisotropic_jacobian()
    test_aristote_trivial()
    test_probe_along_u()
    test_low_rank_random_vs_u()
    test_gradient_only_to_predictor()
    test_phase_model()
    test_lejepa_forward()
    print('\nTous les tests sont passes.')
