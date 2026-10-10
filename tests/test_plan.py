"""Tests du rollout avec contexte (scripts/plan/psc_plan.py), sur CPU.

1. H = 1 : rollout_with_history redonne exactement le rollout de stable-worldmodel 0.1.1.
2. H = 3 : les frames et actions passees comptent (les changer change la prediction).
3. Le cout est differentiable par rapport aux actions candidates (solveur a gradient).

Lancer avec :  python tests/test_plan.py
"""
import os
import sys

import torch

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, '..', 'scripts', 'train'))
sys.path.insert(0, os.path.join(HERE, '..', 'scripts', 'plan'))
sys.path.insert(0, HERE)
from psc_plan import rollout_with_history  # noqa: E402
from test_scal_loss import A, D, tiny_phase_model  # noqa: E402

torch.manual_seed(0)


def models():
    for history in (False, True):
        m = tiny_phase_model(history).eval()
        with torch.no_grad():
            for prm in m.predictor.parameters():
                prm.add_(0.3 * torch.randn_like(prm))
        yield history, m


def test_h1_equals_lib():
    for history, m in models():
        px = torch.randn(2, 4, 1, 3, 8, 8)
        acts = torch.randn(2, 4, 5, A)
        ref = m.rollout({'pixels': px.clone()}, acts.clone())['predicted_emb']
        new = rollout_with_history(m, {'pixels': px.clone()}, acts.clone())['predicted_emb']
        assert ref.shape == new.shape and torch.allclose(ref, new, atol=1e-5), history
    print('H = 1 : identique au rollout de la lib')


def test_h3_uses_past():
    for history, m in models():
        px = torch.randn(2, 4, 3, 3, 8, 8)
        acts = torch.randn(2, 4, 5, A)
        past = torch.randn(2, 4, 2, A)
        out = rollout_with_history(m, {'pixels': px, 'action_history': past}, acts)['predicted_emb']
        assert out.shape == (2, 4, 3 + 5, D)
        out2 = rollout_with_history(m, {'pixels': px, 'action_history': past + 1}, acts)['predicted_emb']
        if history:   # Markov strict : (z_t, u_t, a_t) seulement, les actions passees ne comptent pas
            assert not torch.allclose(out[:, :, 3:], out2[:, :, 3:]), 'actions passees ignorees'
        else:
            assert torch.allclose(out[:, :, 3:], out2[:, :, 3:], atol=1e-5)
        px2 = px.clone()
        px2[:, :, 1] += 1   # z_{t-1} : utilisee par les deux variantes (via u_t)
        out3 = rollout_with_history(m, {'pixels': px2, 'action_history': past}, acts)['predicted_emb']
        assert not torch.allclose(out[:, :, 3:], out3[:, :, 3:]), 'frames passees ignorees'
    print('H = 3 : frames et actions passees prises en compte')


def test_cost_differentiable():
    for history, m in models():
        m.rollout = rollout_with_history.__get__(m)
        # get_cost de la lib suppose un env a la fois (batch_size: 1 dans les configs)
        info = {'pixels': torch.randn(1, 4, 3, 3, 8, 8), 'action_history': torch.randn(1, 4, 2, A),
                'goal': torch.randn(1, 4, 1, 3, 8, 8),
                'action': torch.zeros(1, 4, 1, A)}   # fourni par l'env dans la vraie boucle
        acts = torch.randn(1, 4, 5, A, requires_grad=True)
        m.get_cost(info, acts).sum().backward()
        assert acts.grad is not None and acts.grad.abs().sum() > 0
    print('cout differentiable par rapport aux actions')


if __name__ == '__main__':
    test_h1_equals_lib()
    test_h3_uses_past()
    test_cost_differentiable()
    print('\nTous les tests sont passes.')
