import os
from pathlib import Path

import hydra
import lightning as pl
import stable_pretraining as spt
from stable_pretraining import data as dt
import stable_worldmodel as swm
import torch
from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf, open_dict
from einops import rearrange

from functools import partial
from stable_worldmodel.data import column_normalizer as get_column_normalizer
from stable_worldmodel.wm.loss import SIGReg
from lightning.pytorch.callbacks import Callback
from stable_worldmodel.wm.utils import save_pretrained

from phase_space import MetricsLog, scal_terms

# Metriques par run dans $PSC_RUN_DIR/metrics.jsonl (cf. phase_space.MetricsLog)
METRICS = (
    MetricsLog(os.path.join(os.environ['PSC_RUN_DIR'], 'metrics.jsonl'))
    if os.environ.get('PSC_RUN_DIR')
    else None
)


def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(
        **imagenet_stats, source=source, target=target
    )
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )


def lejepa_forward(self, batch, stage, cfg):
    """encode observations, predict next states, compute losses.

    L_jac (PSC) : regularise le jacobien local d f_theta / d a autour de la
    derniere action de la fenetre de contexte, pour tester l'extension au cas
    non lineaire de l'hypothese implicite du Theoreme 4.4 (B localement
    constant). Voir claude/plan-action-couplage-jacobien-action.md (sketch V2).
    Desactive par defaut (cfg.loss.jac.weight = 0.0).
    """

    ctx_len = cfg.wm.history_size
    n_preds = cfg.wm.num_preds
    lambd = cfg.loss.sigreg.weight
    lambd_jac = cfg.loss.jac.weight
    eps_jac = cfg.loss.jac.eps
    lambd_scal = cfg.loss.scal.weight

    # Replace NaN values with 0 (occurs at sequence boundaries)
    batch['action'] = torch.nan_to_num(batch['action'], 0.0)

    output = self.model.encode(batch)

    emb = output['emb']  # (B, T, D)
    act_emb = output['act_emb']

    ctx_emb = emb[:, :ctx_len]
    ctx_act = act_emb[:, :ctx_len]

    # label : les ctx_len frames suivantes (la fenetre peut etre plus longue
    # quand loss.multi.k > 1 : num_steps = history_size + k)
    tgt_emb = emb[:, n_preds:n_preds + ctx_len]  # label
    pred_emb = self.model.predict(ctx_emb, ctx_act)  # pred

    # Idee 9 (espace des phases) : u_0 = z_0 - z_{-1} est inconnu (u_0 = 0 par
    # convention). Par defaut la position 0 est exclue de L_pred ; avec
    # wm.phase_include_first=true elle est gardee, pour que le modele sache
    # predire sans vitesse connue -- c'est le cas du premier pas de
    # planification (stable-worldmodel 0.1.1 ne donne qu'une frame de contexte).
    # L_scal ne porte jamais sur la position 0.
    phase = getattr(self.model, 'phase', False)   # Idee 9 (global ou spatiale)
    agg = getattr(self.model, 'agg', None)        # tete d'agregation (latent spatial)
    n_patches = getattr(self.model, 'num_patches', None)
    first = 1 if phase and not cfg.wm.get('phase_include_first', False) else 0

    # LeWM loss
    output['pred_loss'] = (pred_emb[:, first:] - tgt_emb[:, first:]).pow(2).mean()
    if n_patches:
        # latent spatial : SIGReg par position de patch (chaque (t, i) = un « temps »
        # de SIGReg, echantillons = le batch), meme echelle que LeWM global
        sig_in = rearrange(emb, 'b t (p d) -> (t p) b d', p=n_patches)
    else:
        sig_in = emb.transpose(0, 1)
    output['sigreg_loss'] = self.sigreg(sig_in)
    total_loss = output['pred_loss'] + lambd * output['sigreg_loss']

    # Metrique comparable entre tous les bras : erreur sur la derniere position
    # (celle utilisee en planification), normalisee par la variance de la cible.
    with torch.no_grad():
        mse_last = (pred_emb[:, -1] - tgt_emb[:, -1]).float().pow(2).mean()
        metrics = {
            'pred_last_mse': mse_last,
            'pred_last_nmse': mse_last / tgt_emb[:, -1].float().var(0).mean(),
        }

    if phase:
        # L_scal au dernier pas de la fenetre : J_u = d h / d u_t, le reste de la
        # fenetre fixe. Sonde de la loss : loss.scal.probe ('random' | 'u'). Les
        # deux sondes sont toujours mesurees (sans gradient pour l'autre), et la
        # loss est calculee meme quand lambd_scal = 0, comme diagnostic.
        model = self.model
        u = model.velocity(ctx_emb)
        probe = cfg.loss.scal.get('probe', 'random')
        terms = {}
        for pr in ('random', 'u'):
            train_it = pr == probe and lambd_scal > 0 and torch.is_grad_enabled()
            with torch.set_grad_enabled(train_it):
                terms[pr] = scal_terms(model, ctx_emb, u, ctx_act,
                                       eps=cfg.loss.scal.eps, probe=pr, agg=agg)
        output['scal_loss'] = terms[probe][cfg.loss.scal.norm].mean()
        if lambd_scal > 0:
            total_loss = total_loss + lambd_scal * output['scal_loss']
        r, v = terms['random'], terms['u']
        metrics.update({
            # sonde aleatoire
            'scal_pdf': r['pdf'].mean(), 'scal_inv': r['inv'].mean(),
            'gamma_mean': r['gamma'].mean(), 'gamma_std': r['gamma'].std(),
            'jw_norm': r['jw'].mean(),
            # sonde le long de u_t : gain, gamma et alignement dans la direction visitee
            'u_scal_inv': v['inv'].mean(), 'u_gamma_mean': v['gamma'].mean(),
            'u_gamma_std': v['gamma'].std(), 'u_jw_norm': v['jw'].mean(),
            'u_cos': (v['gamma'] / v['jw'].clamp_min(1e-12)).mean(),
            'u_norm': u[:, -1].float().norm(dim=-1).mean(),
        })
        # Ablation : h_theta utilise-t-il la vitesse ? Erreur de prediction au
        # dernier pas quand u est remplace par 0, ou par le u d'un autre echantillon.
        with torch.no_grad():
            tgt = tgt_emb[:, -1].float()
            var = tgt.var(0).mean()
            perm = torch.randperm(u.size(0), device=u.device)
            for name, u_alt in (('abl_u0_nmse', torch.zeros_like(u)),
                                ('abl_ushuf_nmse', u[perm])):
                pred = (ctx_emb + model.h(ctx_emb, u_alt, ctx_act))[:, -1].float()
                metrics[name] = (pred - tgt).pow(2).mean() / var

    if lambd_jac > 0:
        # Action brute (pre-encodage) de la derniere position de la fenetre :
        # c'est le a_t de la derivation, celui dont on regularise le jacobien.
        a_last = batch['action'][:, ctx_len - 1]  # (B, action_dim)
        u = torch.randn_like(a_last)
        u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        # action_encoder (Embedder = Conv1d kernel_size=1 + MLP) est ponctuel
        # dans le temps : aucune fuite entre positions, on peut donc ré-encoder
        # uniquement l'action perturbée sans toucher au reste de la fenêtre.
        e_minus = self.model.action_encoder(
            (a_last - eps_jac * u).unsqueeze(1)
        ).squeeze(1)
        e_plus = self.model.action_encoder(
            (a_last + eps_jac * u).unsqueeze(1)
        ).squeeze(1)

        ctx_act_minus = ctx_act.clone()
        ctx_act_minus[:, -1] = e_minus
        ctx_act_plus = ctx_act.clone()
        ctx_act_plus[:, -1] = e_plus

        # Predicteur causal (attention is_causal=True) : seule la derniere
        # position de la sortie depend de la derniere action -> on ne regarde
        # que pred_emb[:, -1] pour un cosinus qui a un sens.
        p0 = pred_emb[:, -1]
        p_minus = self.model.predict(ctx_emb, ctx_act_minus)[:, -1]
        p_plus = self.model.predict(ctx_emb, ctx_act_plus)[:, -1]

        w_minus = p0 - p_minus
        w_plus = p_plus - p0
        cos_jac = torch.nn.functional.cosine_similarity(
            w_minus, w_plus, dim=-1
        )
        output['jac_loss'] = (1 - cos_jac).mean()
        total_loss = total_loss + lambd_jac * output['jac_loss']

    # L_curv (Temporal Straightening, Wang et al. 2026 ; 14/10/2026) : redresse les
    # trajectoires de l'encodeur, 1 - cos(v_t, v_{t+1}) avec v_t = z_{t+1} - z_t, sur
    # toute la fenetre. Poids alpha = loss.curv.weight ; mesuree aussi quand alpha = 0.
    v_enc = emb[:, 1:] - emb[:, :-1]
    if agg is not None:   # latent spatial : sur la tete d'agregation, comme le papier
        v_enc = agg(v_enc)
    cos_v = torch.nn.functional.cosine_similarity(
        v_enc[:, 1:].float(), v_enc[:, :-1].float(), dim=-1)
    output['curv_loss'] = (1 - cos_v).mean()
    alpha_curv = float(cfg.loss.get('curv', {}).get('weight', 0.0))
    if alpha_curv > 0:
        total_loss = total_loss + alpha_curv * output['curv_loss']

    # Loss sur plusieurs blocs (13/10/2026) : a partir des ctx_len vraies frames, le
    # modele enchaine k predictions en reutilisant les siennes (comme le planificateur),
    # avec les vraies actions ; on penalise l'erreur moyenne sur les k blocs, gradient
    # propage a travers toute la chaine. Vise la derive d'integration de l'Idee 9.
    k_multi = int(cfg.loss.get('multi', {}).get('k', 0))
    if k_multi > 1:
        assert emb.size(1) >= ctx_len + k_multi, (
            f'fenetre trop courte : data.dataset.num_steps doit valoir >= {ctx_len + k_multi}')
        roll = list(emb[:, :ctx_len].unbind(1))
        errs = []
        for t in range(k_multi):
            lo = len(roll) - ctx_len
            nxt = self.model.predict(torch.stack(roll[lo:], 1),
                                     act_emb[:, lo:ctx_len + t])[:, -1]
            roll.append(nxt)
            errs.append((nxt - emb[:, ctx_len + t]).pow(2).mean())
        output['multi_loss'] = torch.stack(errs).mean()
        total_loss = total_loss + cfg.loss.multi.weight * output['multi_loss']
        with torch.no_grad():
            for t in (0, k_multi - 1):
                tgt = emb[:, ctx_len + t].float()
                metrics[f'roll_k{t + 1}_nmse'] = (
                    (roll[ctx_len + t].float() - tgt).pow(2).mean() / tgt.var(0).mean())

    output['loss'] = total_loss

    losses_dict = {
        f'{stage}/{k}': v.detach() for k, v in output.items() if 'loss' in k
    }
    self.log_dict(losses_dict, on_step=True, sync_dist=True)

    if METRICS is not None:
        metrics.update({k: v for k, v in output.items() if 'loss' in k})
        METRICS.update(stage, self.current_epoch, self.global_step,
                       {k: v.detach().float().item() for k, v in metrics.items()})
    return output


@hydra.main(version_base=None, config_path='./config', config_name='lewm')
def run(cfg):
    #########################
    ##       dataset       ##
    #########################

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop('name')
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    print(
        f'Loading dataset "{dataset_name}" from {"local cache: " + cache_dir if cache_dir else "default location"}'
    )
    dataset = swm.data.load_dataset(
        dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg
    )
    transforms = [
        get_img_preprocessor(
            source='pixels', target='pixels', img_size=cfg.img_size
        )
    ]

    with open_dict(cfg):
        for col in cfg.data.dataset.keys_to_load:
            if col.startswith('pixels'):
                continue

            normalizer = get_column_normalizer(dataset, col, col)
            transforms.append(normalizer)

        cfg.model.action_encoder.input_dim = (
            cfg.data.dataset.frameskip * dataset.get_dim('action')
        )

    transform = spt.data.transforms.Compose(*transforms)
    dataset.transform = transform

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, val_set = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )

    train = torch.utils.data.DataLoader(
        train_set,
        **cfg.loader,
        generator=rnd_gen,
    )
    val_cfg = {**cfg.loader}
    val_cfg['shuffle'] = False
    val_cfg['drop_last'] = False
    val = torch.utils.data.DataLoader(val_set, **val_cfg)

    ##############################
    ##       model / optim      ##
    ##############################

    world_model = hydra.utils.instantiate(cfg.model)
    if cfg.get('grad_ckpt', False):
        # PSC (13/10/2026) : recalcul des activations de l'encodeur ViT pendant la
        # retropropagation (memoire GPU divisee, ~30 % de calcul en plus, gradients
        # identiques) -- necessaire pour les fenetres de 8 frames (loss.multi.k = 5)
        world_model.encoder.gradient_checkpointing_enable()

    total_steps = cfg.trainer.max_epochs * len(train)
    optimizers = {
        'model_opt': {
            'modules': 'model',
            'optimizer': dict(cfg.optimizer),
            'scheduler': {
                'type': 'LinearWarmupCosineAnnealingLR',
                'warmup_steps': max(1, int(0.01 * total_steps)),
                'max_steps': total_steps,
            },
            'interval': 'epoch',
        },
    }

    data_module = spt.data.DataModule(train=train, val=val)
    world_model = spt.Module(
        model=world_model,
        sigreg=SIGReg(**cfg.loss.sigreg.kwargs),
        forward=partial(lejepa_forward, cfg=cfg),
        optim=optimizers,
    )

    ##########################
    ##       training       ##
    ##########################

    run_id = cfg.get('subdir') or ''
    run_dir = Path(
        swm.data.utils.get_cache_dir(sub_folder='checkpoints'), run_id
    )

    logger = None
    if cfg.wandb.enabled:
        logger = WandbLogger(**cfg.wandb.config)
        logger.log_hyperparams(OmegaConf.to_container(cfg))

    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / 'config.yaml', 'w') as f:
        OmegaConf.save(cfg, f)

    save_ckpt_callback = SaveCkptCallback(
        run_name=cfg.output_model_name,
        cfg=cfg.model,
        epoch_interval=1,
    )

    trainer = pl.Trainer(
        **cfg.trainer,
        callbacks=[save_ckpt_callback],
        num_sanity_val_steps=1,
        logger=logger,
        enable_checkpointing=True,
    )

    ckpt_path = run_dir / f'{cfg.output_model_name}_weights.ckpt'
    manager = spt.Manager(
        trainer=trainer,
        module=world_model,
        data=data_module,
        ckpt_path=ckpt_path if ckpt_path.exists() else None,
    )

    manager()
    if METRICS is not None:
        METRICS.close()
    return


if __name__ == '__main__':
    run()
