import os

from typing import Optional, Union
import torch
import torch.nn.functional as F

import torchvision

from argparse import ArgumentParser

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from data import CustomDataModule
from model import *

class DDPMLightningModule(LightningModule):
    def __init__(self, hparams, *kwargs) -> None:
        super().__init__()
        self.lr = hparams.lr
        self.weight_decay = hparams.weight_decay
        self.num_timesteps = hparams.timesteps
        self.batch_size = hparams.batch_size
        self.model = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
        )

        self.diffusion = GaussianDiffusion(
            self.model,
            image_size=hparams.shape,
            timesteps=hparams.timesteps,   # number of steps
            loss_type='huber' # L1 or L2 or smooth L1
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='common'): 
        image2d = batch["image2d"]
        # noise = default(noise, lambda: torch.randn_like(image2d))
        noise = torch.randn_like(image2d)
        t = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device).long()
        loss = self.diffusion.forward(image2d, t, noise)
        if batch_idx==0:
            samples = self.diffusion.sample(batch_size=self.batch_size)
            viz2d = torch.cat([image2d, samples], dim=-2)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=8, padding=0)
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


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--timesteps", type=int, default=1000, help="timesteps")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--shape", type=int, default=256, help="spatial size of the tensor")
    parser.add_argument("--train_samples", type=int, default=4000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=800, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")
    
    parser.add_argument("--logsdir", type=str, default='logs', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    
    parser.add_argument("--epochs", type=int, default=501, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    parser = Trainer.add_argparse_args(parser)
    
    # Collect the hyper parameters
    hparams = parser.parse_args()
    # Create data module
    
    train_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    train_label2d_folders = [
    ]

    val_image2d_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]

    test_image2d_folders = val_image2d_folders

    datamodule = CustomDataModule(
        train_image2d_folders = train_image2d_folders, 
        val_image2d_folders = val_image2d_folders, 
        test_image2d_folders = test_image2d_folders, 
        train_samples = hparams.train_samples,
        val_samples = hparams.val_samples,
        test_samples = hparams.test_samples,
        batch_size = hparams.batch_size, 
        shape = hparams.shape
    )
    datamodule.setup(seed=hparams.seed)


    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = DDPMLightningModule(
        hparams = hparams
    )
    model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model

     # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename='{epoch:02d}-{validation_loss_epoch:.2f}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=5, 
    )
    lr_callback = LearningRateMonitor(logging_interval='step')
    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.logsdir, log_graph=True)

    # Init model with callbacks
    trainer = Trainer.from_argparse_args(
        hparams, 
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback, 
        ],
        # accumulate_grad_batches=4, 
        strategy="fsdp", #"fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        precision=16,  #if hparams.use_amp else 32,
        # amp_backend='apex',
        # amp_level='O1', # see https://nvidia.github.io/apex/amp.html#opt-levels
        # stochastic_weight_avg=True,
        # auto_scale_batch_size=True, 
        # gradient_clip_val=5, 
        # gradient_clip_algorithm='norm', #'norm', #'value'
        # track_grad_norm=2, 
        # detect_anomaly=True, 
        # benchmark=None, 
        # deterministic=False,
        # profiler="simple",
    )

    trainer.fit(
        model, 
        datamodule,
    )

    # test

    # serve