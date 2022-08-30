import os
import math 
from argparse import ArgumentParser
from typing import Optional 

import torch 
import torch.nn as nn 

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from monai.networks.layers import * #Reshape
from monai.networks.nets import * #UNet, DenseNet121, Generator
from monai.losses import * #DiceLoss

from positional_encodings.torch_encodings import PositionalEncodingPermute1D

class DenoisingDiffusionLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.lr = hparams.lr
        self.shape = hparams.shape
        self.steps = hparams.steps
        self.tdims = hparams.tdims 
        self.cdims = hparams.cdims
        self.logsdir = hparams.logsdir
        self.batch_size = hparams.batch_size
        self.save_hyperparameters()

        self.model = nn.Sequential(
            UNet(
                spatial_dims=2,
                in_channels=16, #self.shape,
                out_channels=16,
                channels=(64, 128, 256, 512, 1024),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                kernel_size=3,
                up_kernel_size=3,
                act=("LeakyReLU", {"inplace": True}),
                dropout=0.5,
                norm=Norm.BATCH,
                # mode="nontrainable",
            ), 
        )

        self.l1loss = nn.L1Loss(reduction="mean")
        steps_tensor = torch.ones(self.batch_size, self.tdims, self.steps)
        self.pos_enc = PositionalEncodingPermute1D(self.tdims)(steps_tensor)

    def forward(self, x, t):
        # Slicing the time index and expand to image shape
        time_tensor = self.pos_enc[..., t].repeat(1, 1, self.shape, self.shape) 
        data_tensor = x 
        pass 
    
    def _common_step(self, batch, batch_idx, stage: Optional[str]="train"):
        image2d = batch["image2d"]
        _device = image2d.device
        step_id = torch.randint(low=0, high=self.steps+1, size=self.batch_size).to(_device) 
        pass 

    def training_step(self, batch, batch_idx):
        pass


        