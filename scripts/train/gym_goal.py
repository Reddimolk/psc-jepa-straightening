"""Environnements Gymnasium classic-control « but à atteindre » pour le PSC.

Pendulum et MountainCarContinuous : deux systèmes où l'inertie domine (la vitesse
fait partie de l'état et persiste), donc un terrain plus favorable à l'Idée 9 que
PushT. stable-worldmodel 0.1.1 fournit ces environnements mais ni dataset, ni
critère de succès vers un but : ce module ajoute

- un rendu direct en 224x224 (identique à la collecte et à l'évaluation),
  sans la flèche de couple de Pendulum (elle afficherait l'action dans l'image) ;
- set_state / set_goal_state (appelés par World.evaluate via les `callables`
  de la config, sur l'env non enveloppé) ;
- le critère de succès : être à la POSITION du but AU MOMENT où les données
  l'atteignent, c'est-à-dire CHECK_STEP = 25 pas après le départ (= le
  goal_offset_steps de l'évaluation). Réalisable par construction (les actions du
  dataset le font), et sensible à l'inertie : il faut arriver au bon endroit au bon
  moment. Le critère « passer par la position à n'importe quel moment » était
  trivial pour ces systèmes oscillants (politique aléatoire : 83 % sur Pendulum,
  67 % sur MountainCar au test du 07/10/2026). La vitesse n'est pas demandée :
  l'image du but ne la contient pas.

Importer ce module enregistre psc/PendulumGoal-v0 et psc/MountainCarGoal-v0.
"""

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.continuous_mountain_car import (
    Continuous_MountainCarEnv,
)
from gymnasium.envs.classic_control.pendulum import PendulumEnv

IMG = 224
CHECK_STEP = 25  # = eval.goal_offset_steps


class _GoalMixin:
    """set_state / set_goal_state, et succès vérifié au pas CHECK_STEP après
    set_goal_state (World.evaluate : reset, puis callables, puis les pas)."""

    def _goal_init(self):
        self.goal_state = None
        self._t = 0

    def set_state(self, state):
        self.state = np.asarray(state, dtype=np.float64).reshape(-1)[:2].copy()

    def set_goal_state(self, goal_state):
        self.goal_state = np.asarray(goal_state, dtype=np.float64).reshape(-1)[:2]
        self._t = 0

    def _pos_error(self):
        raise NotImplementedError

    def _goal_terminated(self):
        if self.goal_state is None:
            return False
        self._t += 1
        return self._t == CHECK_STEP and self._pos_error() < self.TOL


class PendulumGoal(_GoalMixin, PendulumEnv):
    """État (θ, θ̇). Succès : |θ - θ_but| < TOL au pas CHECK_STEP."""

    TOL = 0.15  # rad, ~8.6°

    def __init__(self, render_mode='rgb_array', **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self.screen_dim = IMG
        self._goal_init()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info['state'] = np.asarray(self.state, dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        obs, reward, _, truncated, info = super().step(action)
        info['state'] = np.asarray(self.state, dtype=np.float32).copy()
        return obs, reward, self._goal_terminated(), truncated, info

    def _pos_error(self):
        d = self.state[0] - self.goal_state[0]
        return abs(np.arctan2(np.sin(d), np.cos(d)))

    def render(self):
        last_u, self.last_u = self.last_u, None  # pas de flèche de couple
        try:
            return super().render()
        finally:
            self.last_u = last_u


class MountainCarGoal(_GoalMixin, Continuous_MountainCarEnv):
    """État (x, ẋ). Succès : |x - x_but| < TOL au pas CHECK_STEP. La terminaison
    native (sommet de droite) est désactivée : les épisodes vont jusqu'au bout."""

    TOL = 0.02  # sur une piste de longueur 1.8

    def __init__(self, render_mode='rgb_array', **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self.screen_width = self.screen_height = IMG
        self._goal_init()

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info['state'] = np.asarray(self.state, dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        obs, reward, _, truncated, info = super().step(action)
        info['state'] = np.asarray(self.state, dtype=np.float32).copy()
        return obs, reward, self._goal_terminated(), truncated, info

    def _pos_error(self):
        return abs(self.state[0] - self.goal_state[0])


for _id, _cls in (('psc/PendulumGoal-v0', PendulumGoal),
                  ('psc/MountainCarGoal-v0', MountainCarGoal)):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point=_cls)
