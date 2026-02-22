import os
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import *
from models.rendering import *

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

import torch.nn.functional as F
from models.nerf import nerfw_forward_xyz_color


class NeRFSystem(LightningModule):
    def __init__(self, hparams, root_dir=None): # Add-in root_dir
        super().__init__()
        #self.hparams = hparams updated
        self.save_hyperparameters(hparams)

        self.loss = loss_dict['nerfw'](coef=1)

        self.models_to_train = []
        self.embedding_xyz = PosEmbedding(hparams.N_emb_xyz-1, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.N_emb_dir-1, hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        #------------------------------------------------------------------------
        self.root_dir = root_dir # Add-in root-dir and color prior
        prior_path = os.path.join(root_dir, "dense/sparse/color_priors.npz") \
            if root_dir is not None else None

        if prior_path and os.path.isfile(prior_path):
            data = np.load(prior_path)
            xyz = torch.tensor(data["xyz"], dtype=torch.float32)
            self.rgb_priors = torch.tensor(data["rgb"] / 255.0, dtype=torch.float32)

            # Compute median radius (convert xyz to numpy only for the median)
            r_med = np.median(np.linalg.norm(xyz.numpy(), axis=-1))

            target_radius = 5.0
            scale_factor = r_med / target_radius

            # Scale xyz inside torch
            xyz_scaled = xyz / scale_factor

            self.xyz_priors = xyz_scaled.to(self.device)
            self.rgb_priors = self.rgb_priors.to(self.device)

            print(f"[Color Prior] Loaded {len(self.xyz_priors)} prior points.")
        else:
            print("[Color Prior] No color_priors.npy found — color prior disabled.")
            self.xyz_priors = None
            self.rgb_priors = None



        #------------------------------------------------------------------------

        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            self.models_to_train += [self.embedding_a]
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            self.models_to_train += [self.embedding_t]

        self.nerf_coarse = NeRF('coarse',
                                in_channels_xyz=6*hparams.N_emb_xyz+3,
                                in_channels_dir=6*hparams.N_emb_dir+3)
        self.models = {'coarse': self.nerf_coarse}
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF('fine',
                                  in_channels_xyz=6*hparams.N_emb_xyz+3,
                                  in_channels_dir=6*hparams.N_emb_dir+3,
                                  encode_appearance=hparams.encode_a,
                                  in_channels_a=hparams.N_a,
                                  encode_transient=hparams.encode_t,
                                  in_channels_t=hparams.N_tau,
                                  beta_min=hparams.beta_min)
            self.models['fine'] = self.nerf_fine
        self.models_to_train += [self.models]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            ts[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir}
        if self.hparams.dataset_name == 'phototourism':
            kwargs['img_downscale'] = self.hparams.img_downscale
            kwargs['val_num'] = self.hparams.num_gpus
            kwargs['use_cache'] = self.hparams.use_cache
        elif self.hparams.dataset_name == 'blender':
           # kwargs['img_wh'] = tuple(self.hparams.img_wh)
            kwargs['perturbation'] = self.hparams.data_perturb
            kwargs['img_wh'] = (512, 512)  # Modify the numbers!!

        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

        # Move prior from CPU to GPU Add-in--------------------------
        if self.xyz_priors is not None:
            self.xyz_priors = self.xyz_priors.to(self.device)
            self.rgb_priors = self.rgb_priors.to(self.device)

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=self.hparams.batch_size,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)

    def training_step(self, batch, batch_nb):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        results = self(rays, ts)

        # original loss
        loss_d = self.loss(results, rgbs)
        loss = sum(loss_d.values())

        # ------------------------------------------------------------------
        # Add-in color prior
        if self.xyz_priors is not None:
            # sample random prior points
            idxs = torch.randint(
                0, len(self.xyz_priors),
                (2048,),
                device=self.device
            )

            pts = self.xyz_priors[idxs].to(self.device)
            target_rgb = self.rgb_priors[idxs].to(self.device)

            # canonical RGB prediction (static only)
            pred_rgb = nerfw_forward_xyz_color(
                system=self,
                model=self.models['fine'],
                xyz=pts
            )

            color_prior_loss = F.mse_loss(pred_rgb, target_rgb)

            # small weight
            loss = loss + 0.05 * color_prior_loss

            # logging
            self.log("train/color_prior_loss", color_prior_loss, prog_bar=False)

        # ------------------------------------------------------------------

        with torch.no_grad():
            typ = 'fine' if 'rgb_fine' in results else 'coarse'
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)

        # logging
        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items():
            self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch['ts']
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        ts = ts.squeeze()  # (H*W)

        results = self(rays, ts)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        log = {'val_loss': loss}
        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        if batch_idx == 0:
            if self.hparams.dataset_name == 'phototourism':
                WH = batch['img_wh']
                W, H = WH[0, 0].item(), WH[0, 1].item()
            else:
                #W, H = self.hparams.img_wh
                # blender datatype with actual img w,h
                #W, H = self.val_dataset.img_wh
                 W, H = 512, 512 # Modify!!!
            img = results[f'rgb_{typ}'].view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W))  # (3, H, W)
            stack = torch.stack([img_gt, img, depth])  # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                              stack, self.global_step)

        # --- Compute PSNR and log it properly ---
        psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
        log['val_psnr'] = psnr_

        # Log metrics so Lightning and checkpoint callback can see them
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr_, prog_bar=True, on_step=False, on_epoch=True)

        print("pred:", results[f'rgb_{typ}'].shape, "target:", rgbs.shape)

        # Manual accumulation for epoch-end averaging
        if not hasattr(self, "_val_outputs"):
            self._val_outputs = []
        self._val_outputs.append({'val_loss': loss, 'val_psnr': psnr_})

        return {'val_loss': loss, 'val_psnr': psnr_}

    def on_validation_epoch_end(self):
        """Lightning 2.x replacement for validation_epoch_end."""
        if not hasattr(self, "_val_outputs") or len(self._val_outputs) == 0:
            return

        mean_loss = torch.stack([x['val_loss'] for x in self._val_outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in self._val_outputs]).mean()

        self.log('val/loss', mean_loss)
        self.log('val/psnr', mean_psnr, prog_bar=True)

        # clear stored outputs for next epoch
        self._val_outputs = []


def main(hparams):
    # Modified
    if not hasattr(hparams, "basedir"):
        hparams.basedir = "./logs"
    if not hasattr(hparams, "exp_name"):
        hparams.exp_name = "default_exp"

    system = NeRFSystem(hparams, root_dir=hparams.root_dir) # Add-in root_dir!
    # Modified with dirpath
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join('ckpts', hparams.exp_name),
        filename="{epoch}",
        monitor="val/psnr",
        mode="max",
        save_top_k=-1  # if Lightning complains, change to None
    )

    # Updated to tensorboardlogger
    logger = TensorBoardLogger(
        save_dir=hparams.basedir,
        name=hparams.exp_name
    )

    trainer = Trainer(
        max_epochs=hparams.num_epochs,
        callbacks=[checkpoint_callback],
        logger=logger,
        enable_progress_bar=True,
        accelerator="gpu" if hparams.num_gpus > 0 else "cpu",
        devices=hparams.num_gpus if hparams.num_gpus > 0 else 1,
        strategy="ddp" if hparams.num_gpus > 1 else 'auto',
        num_sanity_val_steps=1,
        benchmark=True,
        profiler="simple" if hparams.num_gpus == 1 else None,
        enable_model_summary=False  # replaces weights_summary=None
    )

    trainer.fit(system, ckpt_path=hparams.ckpt_path)


if __name__ == '__main__':
    hparams = get_opts()
    main(hparams)