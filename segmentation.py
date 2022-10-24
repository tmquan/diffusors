import os

from typing import Optional, Union, List, Dict
import torch
import torch.nn.functional as F

import torchvision

from argparse import ArgumentParser
from natsort import natsorted

from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

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


class PairedDataModule(LightningDataModule):
    def __init__(self, 
        batch_size: int=32,
        train_image_dirs: List[str]=['/data/train/images'],
        train_label_dirs: List[str]=['/data/train/labels'], 
        val_image_dirs: List[str]=['/data/val/images'], 
        val_label_dirs: List[str]=['/data/val/labels'],
        test_image_dirs: List[str]=['/data/test/images'],
        test_label_dirs: List[str]=['/data/test/labels'],
    ):
        """[summary]
        Args:
            batch_size (int, optional): [description]. Defaults to 32.
            train_image_dirs (List[str], optional): [description]. Defaults to ['/data/train/images'].
            train_label_dirs (List[str], optional): [description]. Defaults to ['/data/train/labels'].
            val_image_dirs (List[str], optional): [description]. Defaults to ['/data/val/images'].
            val_label_dirs (List[str], optional): [description]. Defaults to ['/data/val/labels'].
            test_image_dirs (List[str], optional): [description]. Defaults to ['/data/test/images'].
            test_label_dirs (List[str], optional): [description]. Defaults to ['/data/test/labels'].
        """
        super().__init__()
        self.batch_size = batch_size
        self.train_image_dirs = train_image_dirs
        self.train_label_dirs = train_label_dirs
        self.val_image_dirs = val_image_dirs
        self.val_label_dirs = val_label_dirs
        self.test_image_dirs = test_image_dirs
        self.test_label_dirs = test_label_dirs

    def prepare_data(self):
        # download, split, etc...
        # only called on 1 GPU/TPU in distributed
        pass
    
    def glob_dict(
        self, 
        image_dirs: List[str], 
        label_dirs: List[str],
        ext: str='*.png',
    ) -> Dict[str, List[str]]:
        assert image_dirs is not None and label_dirs is not None
        assert len(image_dirs) == len(label_dirs)
        
        # Glob all image files in image_dirs
        image_paths = [Path(folder).rglob(ext) for folder in image_dirs]
        image_files = natsorted([str(path) for path_list in image_paths for path in path_list])

        # Glob all label files in label_dirs
        label_paths = [Path(folder).rglob(ext) for folder in label_dirs]
        label_files = natsorted([str(path) for path_list in label_paths for path in path_list])

        # Check that the number of image and label files match
        print(f'Found {len(image_files)} images and {len(label_files)} labels.')
        assert len(image_files) == len(label_files)

        # Create a dictionary of image and label files
        data_dicts = [
            {"image": image_file,  
             "label": label_file} for image_file, label_file in zip(image_files, label_files)
        ]
        return data_dicts

    def setup(self, seed: int=42, stage: Optional[str]=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)
        self.train_data_dicts = self.glob_dict(self.train_image_dirs, self.train_label_dirs, ext='*.png')
        self.val_data_dicts = self.glob_dict(self.val_image_dirs, self.val_label_dirs, ext='*.png')
        self.test_data_dicts = self.glob_dict(self.test_image_dirs, self.test_label_dirs, ext='*.png')
        

class DDMIDataModule(PairedDataModule):
    def __init__(self, 
        batch_size: int=32,
        shape: int = 256,
        train_image_dirs: List[str]=['/data/train/images'],
        train_label_dirs: List[str]=['/data/train/labels'], 
        val_image_dirs: List[str]=['/data/val/images'], 
        val_label_dirs: List[str]=['/data/val/labels'],
        test_image_dirs: List[str]=['/data/test/images'],
        test_label_dirs: List[str]=['/data/test/labels'],
        keys: List[str]=["image2d", "label2d"], 
        train_samples: int = 4000,
        val_samples: int = 800,
        test_samples: int = 800,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_image_dirs = train_image_dirs
        self.train_label_dirs = train_label_dirs
        self.val_image_dirs = val_image_dirs
        self.val_label_dirs = val_label_dirs
        self.test_image_dirs = test_image_dirs
        self.test_label_dirs = test_label_dirs
        self.train_samples = train_samples
        self.val_samples = val_samples
        self.test_samples = test_samples
        self.shape = shape
        self.keys = keys

    def _shared_dataloader(self, data_dicts, transforms=None, shuffle=True, drop_last=False, num_workers=8):
        dataset = CacheDataset(
            data=data_dicts, 
            cache_rate=1.0, 
            num_workers=num_workers,
            transform=transforms,
        )
        dataloader = DataLoader(
            dataset=dataset, 
            batch_size=self.batch_size, 
            num_workers=num_workers, 
            collate_fn=list_data_collate,
            shuffle=shuffle,
        )
        return dataloader

    def train_dataloader(self):
        train_transforms = Compose(
            [
                LoadImaged(keys=self.keys),
                AddChanneld(keys=self.keys,),
                ScaleIntensityd(keys=self.keys, minv=0.0, maxv=1.0,),
                RandZoomd(keys=self.keys, prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["area"]), 
                Resized(keys=self.keys, spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=self.keys, k=256, mode="constant", constant_values=0),
                ToTensord(keys=self.keys,),
            ]
        )
        return self._shared_dataloader(self.train_data_dicts, 
            transforms=train_transforms, 
            shuffle=True,
            drop_last=False,
            num_workers=4
        )
    
    def val_dataloader(self):
        val_transforms = Compose(
            [
                LoadImaged(keys=self.keys),
                AddChanneld(keys=self.keys,),
                ScaleIntensityd(keys=self.keys, minv=0.0, maxv=1.0,),
                Resized(keys=self.keys, spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=self.keys, k=256, mode="constant", constant_values=0),
                ToTensord(keys=self.keys,),
            ]
        )
        return self._shared_dataloader(self.val_data_dicts, 
            transforms=val_transforms, 
            shuffle=False,
            drop_last=False,
            num_workers=2
        )

    def test_dataloader(self):
        test_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Rotate90d(keys=["image", "label"]),
                Flipd(keys=["image", "label"], spatial_axis=0),
                Resized(
                    keys=["image", "label"], 
                    spatial_size=(512, 512),
                    mode=["bilinear", "nearest"],
                ),
                ScaleIntensityd(
                    keys=["image", "label"], 
                    minv=0.0, 
                    maxv=1.0,
                ),
            ]
        )
        return self._shared_dataloader(self.test_data_dicts, 
            transforms=test_transforms, 
            shuffle=False,
            drop_last=False,
            num_workers=2
        )


class DDMILightningModule(LightningModule):
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

    val_image_dirs = [
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

    val_label_dirs = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/labels/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/labels/'), 
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/labels/'),
    ]

    test_image_dirs = val_image_dirs
    test_label_dirs = val_label_dirs

    datamodule_label = DDMIDataModule(
        train_image_dirs = train_image_dirs, 
        train_label_dirs = train_label_dirs, 
        val_image_dirs = val_image_dirs, 
        val_label_dirs = val_label_dirs, 
        test_image_dirs = test_image_dirs, 
        test_label_dirs = test_label_dirs, 
        train_samples = hparams.train_samples,
        val_samples = hparams.val_samples,
        test_samples = hparams.test_samples,
        batch_size = hparams.batch_size, 
        shape = hparams.shape,
        keys = ["image", "label"]
    )

    datamodule_label.setup(seed=hparams.seed)

    train_image_dirs = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
    ]
    train_label_dirs = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
    ]
    val_image_dirs = train_image_dirs
    val_label_dirs = train_label_dirs
    test_image_dirs = val_image_dirs
    test_label_dirs = val_label_dirs

    datamodule_unsup = DDMIDataModule(
        train_image_dirs = train_image_dirs, 
        train_label_dirs = train_label_dirs, 
        val_image_dirs = val_image_dirs, 
        val_label_dirs = val_label_dirs, 
        test_image_dirs = test_image_dirs, 
        test_label_dirs = test_label_dirs, 
        train_samples = hparams.train_samples,
        val_samples = hparams.val_samples,
        test_samples = hparams.test_samples,
        batch_size = hparams.batch_size, 
        shape = hparams.shape,
        keys = ["image", "label"]
    )

    datamodule_unsup.setup(seed=hparams.seed)
    # ####### Test camera mu and bandwidth ########
    # # test_random_uniform_cameras(hparams, datamodule)
    # #############################################

    # model = DDMILightningModule(
    #     hparams = hparams
    # )
    # model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model

    #  # Seed the application
    # seed_everything(42)

    # # Callback
    # checkpoint_callback = ModelCheckpoint(
    #     dirpath=hparams.logsdir,
    #     filename='{epoch:02d}-{validation_loss_epoch:.2f}',
    #     save_top_k=-1,
    #     save_last=True,
    #     every_n_epochs=5, 
    # )
    # lr_callback = LearningRateMonitor(logging_interval='step')
    # # Logger
    # tensorboard_logger = TensorBoardLogger(save_dir=hparams.logsdir, log_graph=True)

    # # Init model with callbacks
    # trainer = Trainer.from_argparse_args(
    #     hparams, 
    #     max_epochs=hparams.epochs,
    #     logger=[tensorboard_logger],
    #     callbacks=[
    #         lr_callback,
    #         checkpoint_callback, 
    #     ],
    #     # accumulate_grad_batches=4, 
    #     strategy="fsdp", #"fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
    #     precision=16,  #if hparams.use_amp else 32,
    #     # amp_backend='apex',
    #     # amp_level='O1', # see https://nvidia.github.io/apex/amp.html#opt-levels
    #     # stochastic_weight_avg=True,
    #     # auto_scale_batch_size=True, 
    #     # gradient_clip_val=5, 
    #     # gradient_clip_algorithm='norm', #'norm', #'value'
    #     # track_grad_norm=2, 
    #     # detect_anomaly=True, 
    #     # benchmark=None, 
    #     # deterministic=False,
    #     # profiler="simple",
    # )

    # trainer.fit(
    #     model, 
    #     datamodule,
    # )

    # # test

    # # serve