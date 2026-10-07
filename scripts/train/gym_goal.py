"""Environnements Gymnasium classic-control « but à atteindre » pour le PSC.

Pendulum et MountainCarContinuous : deux systèmes où l'inertie domine (la vitesse
fait partie de l'état et persiste), donc un terrain plus favorable à l'Idée 9 que
PushT. stable-worldmodel 0.1.1 fournit ces environnements mais ni dataset, ni
critère de succès vers un but : ce module ajoute

- un rendu direct en 224x224 (identique à la collecte et à l'évaluation),
  sans la flèche de couple de Pendulum (elle afficherait l'action dans l'image) ;
- set_state / set_goal_state (appelés par World.evaluate via les `callables`
  de la config, sur l'env non enveloppé) ;
- terminated = succès = la POSITION du but est atteinte. La vitesse n'est pas
  demandée : l'image du but ne la contient pas, et la plupart des états de
  Pendulum ne sont pas maintenables avec le couple disponible.

Importer ce module enregistre psc/PendulumGoal-v0 et psc/MountainCarGoal-v0.
"""

import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.continuous_mountain_car import (
    Continuous_MountainCarEnv,
)
from gymnasium.envs.classic_control.pendulum import PendulumEnv

IMG = 224


class PendulumGoal(PendulumEnv):
    """État (θ, θ̇). Succès : |θ - θ_but| < TOL (angle ramené dans [-π, π])."""

    TOL = 0.15  # rad, ~8.6°

    def __init__(self, render_mode='rgb_array', **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self.screen_dim = IMG
        self.goal_state = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info['state'] = np.asarray(self.state, dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        info['state'] = np.asarray(self.state, dtype=np.float32).copy()
        if self.goal_state is not None:
            d = self.state[0] - self.goal_state[0]
            terminated = bool(abs(np.arctan2(np.sin(d), np.cos(d))) < self.TOL)
        return obs, reward, terminated, truncated, info

    def set_state(self, state):
        self.state = np.asarray(state, dtype=np.float64).reshape(-1)[:2].copy()

    def set_goal_state(self, goal_state):
        self.goal_state = np.asarray(goal_state, dtype=np.float64).reshape(-1)[:2]

    def render(self):
        last_u, self.last_u = self.last_u, None  # pas de flèche de couple
        try:
            return super().render()
        finally:
            self.last_u = last_u


class MountainCarGoal(Continuous_MountainCarEnv):
    """État (x, ẋ). Succès : |x - x_but| < TOL. La terminaison native (sommet
    de droite) est désactivée : les épisodes de collecte vont jusqu'au bout."""

    TOL = 0.03  # sur une piste de longueur 1.8

    def __init__(self, render_mode='rgb_array', **kwargs):
        super().__init__(render_mode=render_mode, **kwargs)
        self.screen_width = self.screen_height = IMG
        self.goal_state = None

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        info['state'] = np.asarray(self.state, dtype=np.float32).copy()
        return obs, info

    def step(self, action):
        obs, reward, _, truncated, info = super().step(action)
        info['state'] = np.asarray(self.state, dtype=np.float32).copy()
        terminated = False
        if self.goal_state is not None:
            terminated = bool(abs(self.state[0] - self.goal_state[0]) < self.TOL)
        return obs, reward, terminated, truncated, info

    def set_state(self, state):
        self.state = np.asarray(state, dtype=np.float64).reshape(-1)[:2].copy()

    def set_goal_state(self, goal_state):
        self.goal_state = np.asarray(goal_state, dtype=np.float64).reshape(-1)[:2]


for _id, _cls in (('psc/PendulumGoal-v0', PendulumGoal),
                  ('psc/MountainCarGoal-v0', MountainCarGoal)):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point=_cls)
