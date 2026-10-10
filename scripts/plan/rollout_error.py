"""Erreur de rollout en boucle ouverte d'un checkpoint (11/10/2026).

Pour des fenêtres du dataset : 3 frames de contexte (comme à l'entraînement), puis K
prédictions enchaînées (chaque prédiction devient le contexte de la suivante), avec
les VRAIES actions du dataset. Erreur au pas k = ||ẑ_{t+k} - E(o_{t+k})||² moyenne,
divisée par la variance de la cible (nmse, comme pred_last_nmse), où E(o_{t+k}) est
l'encodage de la vraie frame par le même modèle. Mêmes fenêtres pour tous les modèles
(--seed). Références sans modèle : « persistance » (ẑ_{t+k} = z_t) et « vitesse
constante » (ẑ_{t+k} = z_t + k u_t).

Note : les fenêtres sont tirées dans tout le dataset (le découpage train/val de
l'entraînement dépend de la longueur des fenêtres et n'est pas reproductible ici) :
c'est une mesure de l'accumulation d'erreur, pas de la généralisation.

  python rollout_error.py --ckpt <weights.pt> --data pendulum --out rollout.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'train'))
import stable_pretraining as spt  # noqa: E402
import stable_worldmodel as swm  # noqa: E402
from stable_worldmodel.data import column_normalizer  # noqa: E402

HIST, FS = 3, 5


def img_preprocessor(img_size=224):
    st = spt.data.dataset_stats.ImageNet
    to_image = spt.data.transforms.ToImage(**st, source='pixels', target='pixels')
    resize = spt.data.transforms.Resize(img_size, source='pixels', target='pixels')
    return spt.data.transforms.Compose(to_image, resize)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', required=True, help='pendulum | mountaincar (dossier du dataset)')
    p.add_argument('--K', type=int, default=5)
    p.add_argument('--n', type=int, default=4096)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'

    ds = swm.data.load_dataset(f'{a.data}/train.lance', frameskip=FS, num_steps=HIST + a.K,
                               keys_to_load=['pixels', 'action'], keys_to_cache=['action'])
    # mêmes prétraitements qu'à l'entraînement (lewm.py) : images ImageNet 224, action normalisée
    ds.transform = spt.data.transforms.Compose(
        img_preprocessor(), column_normalizer(ds, 'action', 'action'))
    idx = np.random.default_rng(a.seed).choice(len(ds), size=min(a.n, len(ds)), replace=False)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx.tolist()),
                                         batch_size=a.batch, num_workers=6)

    model = swm.wm.utils.load_pretrained(a.ckpt).to(dev).eval()
    sq = {k: [] for k in ('model', 'persist', 'constvel')}
    tgt_all = []
    for batch in loader:
        b = {k: v.to(dev) for k, v in batch.items() if torch.is_tensor(v)}
        b['action'] = torch.nan_to_num(b['action'], 0.0)
        out = model.encode(b)
        z, act = out['emb'].float(), out['act_emb']
        emb = list(z[:, :HIST].unbind(1))
        for t in range(a.K):
            lo = len(emb) - HIST
            emb.append(model.predict(torch.stack(emb[lo:], 1), act[:, lo:HIST + t]).float()[:, -1])
        pred = torch.stack(emb[HIST:], 1)                         # (B, K, D)
        true = z[:, HIST:]
        zt, ut = z[:, HIST - 1:HIST], z[:, HIST - 1:HIST] - z[:, HIST - 2:HIST - 1]
        steps = torch.arange(1, a.K + 1, device=dev).view(1, -1, 1)
        sq['model'].append((pred - true).pow(2).mean(-1).cpu())
        sq['persist'].append((zt - true).pow(2).mean(-1).cpu())
        sq['constvel'].append((zt + steps * ut - true).pow(2).mean(-1).cpu())
        tgt_all.append(true.cpu())
    var = torch.cat(tgt_all).var(0).mean(-1)                      # (K,)
    res = {'ckpt': a.ckpt, 'data': a.data, 'K': a.K, 'n': int(len(idx)), 'seed': a.seed}
    for k, v in sq.items():
        res[f'nmse_{k}'] = (torch.cat(v).mean(0) / var).tolist()   # une valeur par pas k
    Path(a.out).write_text(json.dumps(res))
    print(json.dumps({k: [round(x, 4) for x in v] if isinstance(v, list) else v for k, v in res.items()}))


if __name__ == '__main__':
    main()
