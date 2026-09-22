"""
Test unitaire de la formule L_jac (couplage courbure-action), independant de stable-worldmodel.

Principe : un predicteur f(ctx, a) EXACTEMENT affine en l'action a un jacobien
d(f)/d(a) constant par construction -> L_jac doit etre quasi nul.
Un predicteur avec une vraie non-linearite en l'action (tanh) a un jacobien qui
varie avec a -> L_jac doit etre nettement plus grand.

Si ce script echoue, le bug est dans la formule/l'implementation de L_jac elle-meme,
pas dans stable-worldmodel : ca isole les deux sources d'erreur.

Lancer avec :  python test_jac_loss.py
Dependance :   pip install torch   (wheel precompile, aucun souci d'installation attendu)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


def jac_loss(predict_fn, ctx, act, eps=1e-2, n_directions=8):
    """
    Estimateur de L_jac par differences finies (type Hutchinson/CURE),
    identique a celui du sketch dans lejepa_forward.

    predict_fn(ctx, act) -> tenseur de sortie, meme signature que self.model.predict(ctx_emb, ctx_act)
    ctx : contexte fixe (n'intervient pas dans la perturbation)
    act : action autour de laquelle on regularise le jacobien local
    """
    losses = []
    for _ in range(n_directions):
        u = torch.randn_like(act)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        pred_minus = predict_fn(ctx, act - eps * u)
        pred_0 = predict_fn(ctx, act)
        pred_plus = predict_fn(ctx, act + eps * u)

        w_minus = pred_0 - pred_minus
        w_plus = pred_plus - pred_0

        cos = F.cosine_similarity(w_minus, w_plus, dim=-1)
        losses.append((1 - cos).mean())
    return torch.stack(losses).mean()


class AffinePredictor(nn.Module):
    """f(ctx, a) = W_ctx . ctx + W_act . a + b -- jacobien en a CONSTANT (= W_act)."""

    def __init__(self, d_ctx=8, d_act=4, d_out=8):
        super().__init__()
        self.lin_ctx = nn.Linear(d_ctx, d_out)
        self.lin_act = nn.Linear(d_act, d_out)

    def forward(self, ctx, act):
        return self.lin_ctx(ctx) + self.lin_act(act)


class NonlinearPredictor(nn.Module):
    """f(ctx, a) = W_ctx . ctx + W_act . tanh(3a) + b -- jacobien en a VARIE avec a."""

    def __init__(self, d_ctx=8, d_act=4, d_out=8):
        super().__init__()
        self.lin_ctx = nn.Linear(d_ctx, d_out)
        self.lin_act = nn.Linear(d_act, d_out)

    def forward(self, ctx, act):
        return self.lin_ctx(ctx) + self.lin_act(torch.tanh(3.0 * act))


def main():
    batch, d_ctx, d_act = 256, 8, 4
    ctx = torch.randn(batch, d_ctx)
    act = torch.randn(batch, d_act)

    affine = AffinePredictor(d_ctx, d_act)
    nonlin = NonlinearPredictor(d_ctx, d_act)

    l_affine = jac_loss(affine, ctx, act).item()
    l_nonlin = jac_loss(nonlin, ctx, act).item()

    print(f"L_jac (predicteur affine, doit etre ~0)      : {l_affine:.3e}")
    print(f"L_jac (predicteur non lineaire, doit etre >0) : {l_nonlin:.3e}")
    print(f"ratio non-lineaire / affine                   : {l_nonlin / max(l_affine, 1e-30):.1f}")

    assert l_affine < 1e-4, (
        f"L_jac devrait etre quasi nul pour un predicteur affine, obtenu {l_affine}"
    )
    assert l_nonlin > 100 * l_affine, (
        "L_jac devrait etre nettement plus grand pour un predicteur non lineaire "
        f"(affine={l_affine}, nonlineaire={l_nonlin})"
    )

    # Verifie aussi que le gradient remonte correctement (pas de graphe casse par un detach oublie)
    ctx_g = ctx.clone().requires_grad_(True)
    act_g = act.clone().requires_grad_(True)
    loss = jac_loss(nonlin, ctx_g, act_g)
    loss.backward()

    assert act_g.grad is not None and torch.isfinite(act_g.grad).all(), (
        "Le gradient de L_jac par rapport a l'action ne remonte pas correctement"
    )
    for name, p in nonlin.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), (
            f"Le gradient de L_jac ne remonte pas jusqu'au parametre {name}"
        )

    print("\nTous les tests sont passes.")


if __name__ == "__main__":
    main()
