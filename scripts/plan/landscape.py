"""Figures 2 et 4 de « Temporal Straightening for Latent Planning », pour nos modèles.

Figure 4 (paysage de coût dans l'espace des actions) : pour un problème départ/but
tiré du dataset, on fixe les DEUX PREMIERS blocs d'actions (nos actions sont 1D, celles
de PushT sont 2D : on prend donc comme axes la valeur, constante sur le bloc, des blocs
1 et 2) sur une grille G x G, on optimise les blocs restants par gradient (K
initialisations, Adam) pour minimiser le coût terminal ||ẑ_H - z_but||², et on trace
le coût minimal atteignable. Plus le paysage est convexe, plus la planification est facile.

Figure 2 : trajectoire latente d'un épisode (une image tous les `frameskip` pas)
projetée sur ses 2 premières composantes principales, et distance (MSE) à la dernière
image au cours du temps.

  python landscape.py --ckpt <weights.pt> --data pendulum --out <dossier> [--grid 21 --inits 16]

Le contexte donné au modèle est de 2 images (vitesse connue), comme en évaluation.
Mêmes problème et épisode pour tous les modèles (--seed).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from sklearn import preprocessing  # noqa: E402
from stable_worldmodel.data.formats.lance import LanceDataset  # noqa: E402
from torchvision.transforms import v2 as T  # noqa: E402

import gym_goal  # noqa: E402,F401
import psc_plan  # noqa: E402

FS, H, HORIZON, OFFSET = 5, 2, 5, 25
AMAX = {'pendulum': 2.0, 'mountaincar': 1.0}
MIN_GOAL = {'pendulum': 0.5, 'mountaincar': 0.1}


def img_tf(dev):
    tf = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True),
                    T.Normalize(**spt.data.dataset_stats.ImageNet), T.Resize(224)])
    return lambda px: torch.stack([tf(np.asarray(p)) for p in px]).to(dev)


def pick_problem(ds, env, rng):
    """Un départ (avec 1 bloc de passé) et le but OFFSET pas plus loin, non trivial."""
    ep = ds.get_col_data('episode_idx')
    st = ds.get_col_data('step_idx')
    s0 = ds.get_col_data('state')[:, 0].astype(np.float64)
    while True:
        i = int(rng.integers(0, len(ep) - OFFSET - 1))
        if st[i] < FS or ep[i + OFFSET] != ep[i]:
            continue
        d = s0[i + OFFSET] - s0[i]
        if env == 'pendulum':
            d = np.arctan2(np.sin(d), np.cos(d))
        if abs(d) >= MIN_GOAL[env]:
            return int(ep[i]), int(st[i])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', required=True, choices=sorted(AMAX))
    p.add_argument('--out', required=True)
    p.add_argument('--grid', type=int, default=21)
    p.add_argument('--inits', type=int, default=16)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--chunk', type=int, default=1024, help='sequences optimisees a la fois')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--datasets', default=None, help='dossier des datasets (défaut : $STABLEWM_HOME/datasets)')
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    root = a.datasets or str(swm.data.utils.get_cache_dir(sub_folder='datasets'))
    ds = LanceDataset(f'{root}/{a.data}/train.lance'.replace('\\', '/'),
                      keys_to_load=['pixels', 'action', 'state'], keys_to_cache=['action', 'state'])
    scaler = preprocessing.StandardScaler().fit(ds.get_col_data('action'))
    norm = lambda x: (np.asarray(x, dtype=np.float32) - scaler.mean_[0]) / scaler.scale_[0]
    tf = img_tf(dev)

    model = swm.wm.utils.load_pretrained(a.ckpt).to(dev).eval()
    model.requires_grad_(False)
    psc_plan.install_history_rollout(model)

    rng = np.random.default_rng(a.seed)
    ep, s = pick_problem(ds, a.data, rng)
    ch = ds.load_chunk(np.array([ep]), np.array([s - FS]), np.array([s + OFFSET + 1]))[0]
    px = [np.asarray(f.permute(1, 2, 0)) for f in ch['pixels']]
    act = np.asarray(ch['action'], dtype=np.float32).reshape(-1)
    ctx = tf([px[0], px[FS]]).unsqueeze(0)                      # (1, H, C, h, w)
    goal = tf([px[FS + OFFSET]]).unsqueeze(0)                   # (1, 1, C, h, w)
    hist = torch.tensor(norm(act[:FS]), device=dev).view(1, 1, 1, FS)

    with torch.no_grad():
        goal_emb = model.encode({'pixels': goal})['emb'][:, -1]  # (1, D)
        ctx_emb = model.encode({'pixels': ctx})['emb']           # (1, H, D)

    # ---- Figure 4 : paysage de coût
    lo, hi = norm(-AMAX[a.data]), norm(AMAX[a.data])
    vals = np.linspace(-AMAX[a.data], AMAX[a.data], a.grid)
    g1, g2 = np.meshgrid(vals, vals, indexing='ij')
    fixed_all = torch.tensor(np.stack([norm(g1.ravel()), norm(g2.ravel())], -1), device=dev)
    fixed_all = fixed_all.repeat_interleave(a.inits, 0).unsqueeze(-1).expand(-1, 2, FS)   # (N, 2, FS)
    free_all = torch.randn(fixed_all.size(0), HORIZON - 2, FS, device=dev,
                           generator=torch.Generator(dev).manual_seed(a.seed)).clamp(lo, hi)

    def cost(fixed, free_acts):
        n = fixed.size(0)
        seq = torch.cat([fixed, free_acts], 1).unsqueeze(0)     # (1, n, HORIZON, FS)
        info = {'pixels': ctx.unsqueeze(1).expand(1, n, *ctx.shape[1:]),
                'action_history': hist.expand(1, n, 1, FS), 'emb': ctx_emb.unsqueeze(1).expand(1, n, -1, -1)}
        pred = model.rollout(info, seq)['predicted_emb'][0, :, -1]   # (n, D)
        return (pred - goal_emb).pow(2).sum(-1)

    # par morceaux de --chunk sequences (7 056 d'un coup depassent la memoire GPU)
    costs = []
    for i in range(0, fixed_all.size(0), a.chunk):
        fixed = fixed_all[i:i + a.chunk]
        free = free_all[i:i + a.chunk].clone().requires_grad_(True)
        opt = torch.optim.Adam([free], lr=a.lr)
        for _ in range(a.steps):
            opt.zero_grad()
            cost(fixed, free).sum().backward()
            opt.step()
            with torch.no_grad():
                free.clamp_(lo, hi)
        with torch.no_grad():
            costs.append(cost(fixed, free))
    c = torch.cat(costs).view(a.grid, a.grid, a.inits).min(-1).values.cpu().numpy()

    # ---- Figure 2 : trajectoire latente de l'épisode
    full = ds.load_chunk(np.array([ep]), np.array([0]), np.array([int(ds.lengths[ep])]))[0]
    frames = [np.asarray(f.permute(1, 2, 0)) for f in full['pixels']][::FS]
    with torch.no_grad():
        z = torch.cat([model.encode({'pixels': tf(frames[i:i + 32]).unsqueeze(0)})['emb'][0]
                       for i in range(0, len(frames), 32)]).float().cpu().numpy()
    zc = z - z.mean(0)
    _, _, vt = np.linalg.svd(zc, full_matrices=False)
    pc = zc @ vt[:2].T
    mse_goal = ((z - z[-1]) ** 2).mean(-1)
    v = np.diff(z, axis=0)
    cos = (v[1:] * v[:-1]).sum(-1) / (np.linalg.norm(v[1:], axis=-1) * np.linalg.norm(v[:-1], axis=-1) + 1e-12)

    res = {'ckpt': a.ckpt, 'data': a.data, 'episode': ep, 'start': s, 'grid_values': vals.tolist(),
           'landscape': c.tolist(), 'landscape_min': float(c.min()),
           'pca': pc.tolist(), 'mse_to_last': mse_goal.tolist(), 'mean_cos_consecutive': float(cos.mean())}
    (out / 'landscape.json').write_text(json.dumps(res))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    im = ax[0].imshow(c.T, origin='lower', cmap='hot', extent=[vals[0], vals[-1], vals[0], vals[-1]], aspect='auto')
    ax[0].set_xlabel('action du bloc 1')
    ax[0].set_ylabel('action du bloc 2')
    ax[0].set_title('coût terminal minimal (sombre = bas)')
    fig.colorbar(im, ax=ax[0])
    ax[1].plot(pc[:, 0], pc[:, 1], '-o', ms=3)
    ax[1].plot(pc[-1, 0], pc[-1, 1], '*', ms=16, color='gold', mec='k')
    ax[1].set_title(f'ACP de la trajectoire latente (cos moyen {cos.mean():.2f})')
    ax[2].plot(mse_goal)
    ax[2].set_title('MSE à la dernière image')
    ax[2].set_xlabel('temps (blocs)')
    fig.suptitle(Path(a.ckpt).parent.name)
    fig.tight_layout()
    fig.savefig(out / 'landscape.png', dpi=110)
    print(json.dumps({'landscape_min': res['landscape_min'], 'mean_cos': res['mean_cos_consecutive'],
                      'episode': ep, 'start': s}))


if __name__ == '__main__':
    main()
