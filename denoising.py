import os
import json
import argparse

from typing import Optional, Union
import torch
import torch.nn.functional as F

import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from pytorch_lightning import LightningModule

from data import CustomDatamodule
from model import *

class DDPMLightningModule(LightningModule):
    def __init__(self, hparams, *kwargs) -> None:
        super().__init__()
        self.num_timesteps = hparams.timesteps
        self.batch_size = hparams.batch_size
        self.model = Unet(
            dim = 64,
            dim_mults = (1, 2, 4, 8)
        )

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size = hparams.shape,
            timesteps = hparams.timesteps,   # number of steps
            loss_type = 'huber' # L1 or L2 or smooth L1
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='common'): 
        image2d = batch["image2d"]
        noise = default(noise, lambda: torch.randn_like(image2d))

        t = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device).long()
        loss = self.diffusion.forward(image2d, t, noise)
        if batch_idx==0:
            samples = self.diffusion.sample(batch_size=self.batch_size)
            viz2d = torch.cat([image2d, samples], dim=-1)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
        info = {"loss": loss}
        return info

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        return self._common_step(batch, batch_idx, optimizer_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='test')

    def _common_epoch_end(self, outputs, stage: Optional[str]='common'):
        loss = torch.stack([x[f'loss'] for x in outputs]).mean()
        self.log(f'{stage}_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def train_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='validation')
    
    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='test')


