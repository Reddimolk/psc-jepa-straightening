"""Collecte d'un dataset Lance pour psc/PendulumGoal-v0 ou psc/MountainCarGoal-v0.

  python scripts/data/collect_gym.py pendulum   <sortie.lance> --episodes 1500
  python scripts/data/collect_gym.py mountaincar <sortie.lance> --episodes 1500

Politique de collecte : bruit d'Ornstein-Uhlenbeck sur l'action (corrélé dans le
temps, rho = 0.9) pour produire de vrais mouvements avec élan, plutôt qu'un bruit
blanc qui se moyenne. États initiaux tirés largement pour couvrir l'espace des
états. Colonnes : pixels (224x224, JPEG), action (1), state (2) ; la pixel t est
l'observation AVANT l'action t (convention LeWM). Épisodes collectés en parallèle
(un processus par cœur), écrits par un seul LanceWriter.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'train'))
import gym_goal  # noqa: E402

CFG = {
    'pendulum': dict(cls=gym_goal.PendulumGoal, amax=2.0,
                     init=lambda r: [r.uniform(-np.pi, np.pi), r.uniform(-3, 3)]),
    'mountaincar': dict(cls=gym_goal.MountainCarGoal, amax=1.0,
                        init=lambda r: [r.uniform(-1.2, 0.5), r.uniform(-0.05, 0.05)]),
}


def episode(args):
    env_name, seed, length, rho = args
    c = CFG[env_name]
    rng = np.random.default_rng(seed)
    env = c['cls']()
    env.reset(seed=int(seed))
    env.set_state(c['init'](rng))
    pixels, actions, states = [], [], []
    a = rng.uniform(-1, 1) * c['amax']
    for _ in range(length):
        pixels.append(env.render())
        states.append(np.asarray(env.state, dtype=np.float32).copy())
        a = rho * a + np.sqrt(1 - rho ** 2) * rng.normal() * c['amax']
        a = float(np.clip(a, -c['amax'], c['amax']))
        actions.append([a])
        env.step(np.array([a], dtype=np.float32))
    env.close()
    # LanceWriter attend une liste de valeurs par pas de temps
    return {'pixels': [p.astype(np.uint8) for p in pixels],
            'action': [np.asarray(x, dtype=np.float32) for x in actions],
            'state': states}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('env', choices=sorted(CFG))
    p.add_argument('out')
    p.add_argument('--episodes', type=int, default=1500)
    p.add_argument('--length', type=int, default=200)
    p.add_argument('--rho', type=float, default=0.9)
    p.add_argument('--workers', type=int, default=os.cpu_count())
    a = p.parse_args()

    from stable_worldmodel.data.formats.lance import LanceWriter
    jobs = [(a.env, 1000 + i, a.length, a.rho) for i in range(a.episodes)]
    with Pool(a.workers) as pool, LanceWriter(a.out, mode='overwrite') as w:
        def eps():
            for i, ep in enumerate(pool.imap(episode, jobs, chunksize=4)):
                if (i + 1) % 100 == 0:
                    print(f'{i + 1}/{a.episodes} épisodes', flush=True)
                yield ep
        w.write_episodes(eps())
    print('OK', a.out)


if __name__ == '__main__':
    main()
