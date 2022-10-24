import os
import glob

from typing import Optional, Union, List, Dict, Sequence, Callable
import torch
import torch.nn.functional as F

import torchvision

from argparse import ArgumentParser

from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

import monai 
from monai.data import Dataset, CacheDataset, DataLoader
from monai.data import list_data_collate, decollate_batch
from monai.utils import first, set_determinism, get_seed, MAX_SEED
from monai.transforms import (
    apply_transform, 
    AddChanneld,
    Compose, OneOf, 
    LoadImaged, Spacingd, Lambdad,
    Orientationd, DivisiblePadd, 
    RandFlipd, RandZoomd, RandScaleCropd, CropForegroundd,
    RandAffined,
    Resized, Rotate90d, 
    ScaleIntensityd,
    ScaleIntensityRanged, 
    ToTensord,
)
# from data import CustomDataModule
from model import *

class PairedAndUnsupervisedDataset(monai.data.Dataset, monai.transforms.Randomizable):
    def __init__(
        self,
        keys: Sequence, 
        data: Sequence, 
        transform: Optional[Callable] = None,
        length: Optional[Callable] = None, 
        batch_size: int = 32, 

    ) -> None:
        self.keys = keys
        self.data = data
        self.length = length
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self) -> int:
        if self.length is None:
            return min((len(dataset) for dataset in self.data))
        else: 
            return self.length

    def _transform(self, index: int):
        data = {}
        self.R.seed(index)
        # for key, dataset in zip(self.keys, self.data):
        #     rand_idx = self.R.randint(0, len(dataset)) 
        #     data[key] = dataset[rand_idx]
        rand_idx = self.R.randint(0, len(self.data[0])) 
        data[self.keys[0]] = self.data[0][rand_idx] # image
        data[self.keys[1]] = self.data[1][rand_idx] # label
        rand_idy = self.R.randint(0, len(self.data[2])) 
        data[self.keys[2]] = self.data[2][rand_idy] # unsup

        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data

class PairedAndUnsupervisedDataModule(LightningDataModule):
    def __init__(self, 
        train_image_dirs: str = "path/to/dir", 
        train_label_dirs: str = "path/to/dir", 
        train_unsup_dirs: str = "path/to/dir", 
        val_image_dirs: str = "path/to/dir", 
        val_label_dirs: str = "path/to/dir", 
        val_unsup_dirs: str = "path/to/dir", 
        test_image_dirs: str = "path/to/dir", 
        test_label_dirs: str = "path/to/dir", 
        test_unsup_dirs: str = "path/to/dir", 
        shape: int = 256,
        batch_size: int = 32, 
        train_samples: int = 4000,
        val_samples: int = 800,
        test_samples: int = 800,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shape = shape
        # self.setup() 
        self.train_image_dirs = train_image_dirs
        self.train_label_dirs = train_label_dirs
        self.train_unsup_dirs = train_unsup_dirs
        self.val_image_dirs = val_image_dirs
        self.val_label_dirs = val_label_dirs
        self.val_unsup_dirs = val_unsup_dirs
        self.test_image_dirs = test_image_dirs
        self.test_label_dirs = test_label_dirs
        self.test_unsup_dirs = test_unsup_dirs
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples

        # self.setup()
        def glob_files(folders: str=None, extension: str='*.nii.gz'):
            assert folders is not None
            paths = [glob.glob(os.path.join(folder, extension), recursive = True) for folder in folders]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files
            
        self.train_image_files = glob_files(folders=train_image_dirs, extension='**/*.png')
        self.train_label_files = glob_files(folders=train_label_dirs, extension='**/*.png')
        self.train_unsup_files = glob_files(folders=train_unsup_dirs, extension='**/*.png')
        self.val_image_files = glob_files(folders=val_image_dirs, extension='**/*.png')
        self.val_label_files = glob_files(folders=val_label_dirs, extension='**/*.png')
        self.val_unsup_files = glob_files(folders=val_unsup_dirs, extension='**/*.png')
        self.test_image_files = glob_files(folders=test_image_dirs, extension='**/*.png')
        self.test_label_files = glob_files(folders=test_label_dirs, extension='**/*.png')
        self.test_unsup_files = glob_files(folders=test_unsup_dirs, extension='**/*.png')


    def setup(self, seed: int=42, stage: Optional[str]=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label", "unsup"]),
                AddChanneld(keys=["image", "label", "unsup"],),
                ScaleIntensityd(keys=["image", "label", "unsup"], minv=0.0, maxv=1.0,),
                RandZoomd(keys=["image", "label", "unsup"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["area", "area", "area"]), 
                Resized(keys=["image", "label", "unsup"], spatial_size=256, size_mode="longest", mode=["area", "area", "area"]),
                DivisiblePadd(keys=["image", "label", "unsup"], k=256, mode="constant", constant_values=0),
                ToTensord(keys=["image", "label", "unsup"],),
            ]
        )

        self.train_datasets = PairedAndUnsupervisedDataset(
            keys=["image", "label", "unsup"],
            data=[self.train_image_files, self.train_label_files, self.train_unsup_files], 
            transform=self.train_transforms,
            length=self.train_samples,
            batch_size=self.batch_size,
        )

        self.train_loader = DataLoader(
            self.train_datasets, 
            batch_size=self.batch_size, 
            num_workers=8, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label", "unsup"]),
                AddChanneld(keys=["image", "label", "unsup"],),
                ScaleIntensityd(keys=["image", "label", "unsup"], minv=0.0, maxv=1.0,),
                Resized(keys=["image", "label", "unsup"], spatial_size=256, size_mode="longest", mode=["area", "area", "area"]),
                DivisiblePadd(keys=["image", "label", "unsup"], k=256, mode="constant", constant_values=0),
                ToTensord(keys=["image", "label", "unsup"],),
            ]
        )

        self.val_datasets = PairedAndUnsupervisedDataset(
            keys=["image", "label", "unsup"],
            data=[self.val_image_files, self.val_label_files, self.val_unsup_files], 
            transform=self.val_transforms,
            length=self.val_samples,
            batch_size=self.batch_size,
        )
        
        self.val_loader = DataLoader(
            self.val_datasets, 
            batch_size=self.batch_size, 
            num_workers=4, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader

class DDMILightningModule(LightningModule):
    def __init__(self, hparams, *kwargs) -> None:
        super().__init__()
        self.lr = hparams.lr
        self.weight_decay = hparams.weight_decay
        self.num_timesteps = hparams.timesteps
        self.batch_size = hparams.batch_size
        self.model_image = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
        )

        self.model_label = Unet(
            dim=64,
            dim_mults=(1, 2, 4, 8),
            channels=1,
        )

        self.diffusion_image = GaussianDiffusion(
            self.model_image,
            image_size=hparams.shape,
            timesteps=hparams.timesteps,   # number of steps
            loss_type='L1', # L1 or L2 or smooth L1, 
            objective='pred_x0',
        )

        self.diffusion_label = GaussianDiffusion(
            self.model_label,
            image_size=hparams.shape,
            timesteps=hparams.timesteps,   # number of steps
            loss_type='L1', # L1 or L2 or smooth L1
            objective='pred_x0',
        )

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str]='common'): 
        image, label, unsup = batch["image"], batch["label"], batch["unsup"]

        # noise_p = torch.randn_like(image)
        # t_p = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device).long()
        # loss_image_p = self.diffusion_image.forward(image, t_p, noise_p)
        # loss_label_p = self.diffusion_label.forward(label, t_p, noise_p)

        # noise_u = torch.randn_like(unsup)
        # t_u = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device).long()
        # loss_unsup_u = self.diffusion_image.forward(unsup, t_u, noise_u)
        
        # loss = loss_image_p + loss_label_p + loss_unsup_u

        noise = torch.randn_like(image)
        t = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device).long()
        loss_image = self.diffusion_image.forward(torch.cat([image, unsup], dim=0), 
                                                  torch.cat([t, t], dim=0), 
                                                  torch.cat([noise, noise], dim=0))
        loss_label = self.diffusion_label.forward(label, t, noise)        
        loss = loss_image + loss_label                       
        info = {"loss": loss} 

        if batch_idx==0:
            noise_samples = torch.randn_like(unsup)
            image_samples = self.diffusion_image.sample(batch_size=self.batch_size, img=noise_samples)
            label_samples = self.diffusion_label.sample(batch_size=self.batch_size, img=noise_samples)
            viz2d = torch.cat([image, label, image_samples, label_samples], dim=-2)
            grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=8, padding=0)
            tensorboard = self.logger.experiment
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
        info = {"loss": loss}
        return info

        # noise = torch.randn_like(image2d)
        # t = torch.randint(0, self.num_timesteps, (self.batch_size,), device=self.device).long()
        # loss = self.diffusion.forward(image2d, t, noise)
        # if batch_idx==0:
        #     samples = self.diffusion.sample(batch_size=self.batch_size)
        #     viz2d = torch.cat([image2d, samples], dim=-2)
        #     grid = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=8, padding=0)
        #     tensorboard = self.logger.experiment
        #     tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)
        # info = {"loss": loss}
        # return info

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
    parser.add_argument("--seed", type=int, default=42)
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
    
    train_image_dirs = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62020/20200501/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/T62021/20211101/raw/images'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/train/images/'), 
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    train_label_dirs = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/labels/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/labels/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/labels/'),
    ]

    train_unsup_dirs = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
    ]

    val_image_dirs = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
    ]

    val_label_dirs = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/labels/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/labels/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/labels/'),
    ]

    val_unsup_dirs = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
    ]
    test_image_dirs = val_image_dirs
    test_label_dirs = val_label_dirs
    test_unsup_dirs = val_unsup_dirs

    datamodule = PairedAndUnsupervisedDataModule(
        train_image_dirs = train_image_dirs, 
        train_label_dirs = train_label_dirs, 
        train_unsup_dirs = train_unsup_dirs, 
        val_image_dirs = val_image_dirs, 
        val_label_dirs = val_label_dirs, 
        val_unsup_dirs = val_unsup_dirs, 
        test_image_dirs = test_image_dirs, 
        test_label_dirs = test_label_dirs, 
        test_unsup_dirs = test_unsup_dirs, 
        train_samples = hparams.train_samples,
        val_samples = hparams.val_samples,
        test_samples = hparams.test_samples,
        batch_size = hparams.batch_size, 
        shape = hparams.shape,
        # keys = ["image", "label", "unsup"]
    )

    datamodule.setup(seed=hparams.seed)

    # debug_data = first(datamodule.val_dataloader())
    # image, label, unsup = debug_data["image"], \
    #                       debug_data["label"], \
    #                       debug_data["unsup"]
    # print(image.shape, label.shape, unsup.shape)
    
    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = DDMILightningModule(
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