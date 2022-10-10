import os
import glob
import numpy as np

from typing import Callable, Optional, Sequence

from argparse import ArgumentParser

import monai 
from monai.data import Dataset, DataLoader
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
from pytorch_lightning import LightningDataModule

class UnpairedDataset(Dataset, monai.transforms.Randomizable):
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
        for key, dataset in zip(self.keys, self.data):
            rand_idx = self.R.randint(0, len(dataset)) 
            data[key] = dataset[rand_idx]
        
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data

def HU2Density(hu_values, smoothAir=False):
    # Use two linear interpolations from data: (HU,g/cm^3)
    # -1000 0.00121000000000000
    # -98    0.930000000000000
    # -97    0.930486000000000
    # 14 1.03000000000000
    # 23 1.03100000000000
    # 100    1.11990000000000
    # 101    1.07620000000000
    # 1600   1.96420000000000
    # 3000   2.80000000000000
    # use fit1 for lower HU: density = 0.001029*HU + 1.030 (fit to first 4)
    # use fit2 for upper HU: density = 0.0005886*HU + 1.03 (fit to last 5)

    # set air densities
    if smoothAir:
        hu_values[hu_values <= -900] = -1000
    #hu_values[hu_values > 600] = 5000;
    densities = np.maximum(np.minimum(
        0.001029 * hu_values + 1.030, 0.0005886 * hu_values + 1.03), 0)
    return densities

class CustomDataModule(LightningDataModule):
    def __init__(self, 
        train_image3d_folders: str = "path/to/folder", 
        train_image2d_folders: str = "path/to/folder", 
        val_image3d_folders: str = "path/to/folder", 
        val_image2d_folders: str = "path/to/folder", 
        test_image3d_folders: str = "path/to/folder", 
        test_image2d_folders: str = "path/to/dir", 
        shape: int = 256,
        batch_size: int = 32
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shape = shape
        # self.setup() 
        self.train_image3d_folders = train_image3d_folders
        self.train_image2d_folders = train_image2d_folders
        self.val_image3d_folders = val_image3d_folders
        self.val_image2d_folders = val_image2d_folders
        self.test_image3d_folders = test_image3d_folders
        self.test_image2d_folders = test_image2d_folders

        # self.setup()
        def glob_files(folders: str=None, extension: str='*.nii.gz'):
            assert folders is not None
            paths = [glob.glob(os.path.join(folder, extension), recursive = True) for folder in folders]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files
            
        self.train_image3d_files = glob_files(folders=train_image3d_folders, extension='**/*.nii.gz')
        self.train_image2d_files = glob_files(folders=train_image2d_folders, extension='**/*.png')
        
        self.val_image3d_files = glob_files(folders=val_image3d_folders, extension='**/*.nii.gz') # TODO
        self.val_image2d_files = glob_files(folders=val_image2d_folders, extension='**/*.png')
        
        self.test_image3d_files = glob_files(folders=test_image3d_folders, extension='**/*.nii.gz') # TODO
        self.test_image2d_files = glob_files(folders=test_image2d_folders, extension='**/*.png')


    def setup(self, seed: int=2222, stage: Optional[str]=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image3d", "image2d"]),
                AddChanneld(keys=["image3d", "image2d"],),
                Spacingd(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"]),  
                Rotate90d(keys=["image2d"], k=3),
                OneOf([
                    Orientationd(keys=('image3d'), axcodes="ARI"),
                    Orientationd(keys=('image3d'), axcodes="PRI"),
                    Orientationd(keys=('image3d'), axcodes="ALI"),
                    Orientationd(keys=('image3d'), axcodes="PLI"),
                    Orientationd(keys=["image3d"], axcodes="LAI"),
                    Orientationd(keys=["image3d"], axcodes="RAI"),
                    Orientationd(keys=["image3d"], axcodes="LPI"),
                    Orientationd(keys=["image3d"], axcodes="RPI"),
                    ],
                ),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                #         a_min=-500, #-200, 
                #         a_max=3071, #1500,
                #         b_min=0.0,
                #         b_max=1.0),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                #         a_min=-200, 
                #         a_max=1500,
                #         b_min=0.0,
                #         b_max=1.0),
                ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                        a_min=-500, #-200, 
                        a_max=3071, #1500,
                        b_min=-500,
                        b_max=3071),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                #         a_min=-200, 
                #         a_max=1500,
                #         b_min=-200,
                #         b_max=1500),
                Lambdad(keys=["image3d"], func=HU2Density),
                ScaleIntensityd(keys=["image3d"], 
                        minv=0.0,
                        maxv=1.0),
                RandFlipd(keys=["image3d"], prob=0.5, spatial_axis=0),
                RandZoomd(keys=["image3d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["trilinear"], align_corners=True), 
                RandZoomd(keys=["image2d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["area"]), 
                RandFlipd(keys=["image2d"], prob=0.5, spatial_axis=1),
                RandScaleCropd(keys=["image3d"], 
                               roi_scale=(0.9, 0.9, 0.8), 
                               max_roi_scale=(1.0, 1.0, 0.8), 
                               random_center=True, 
                               random_size=True),
                RandAffined(keys=["image3d"], rotate_range=None, shear_range=None, translate_range=20, scale_range=None),
                Resized(keys=["image3d"], spatial_size=256, size_mode="longest", mode=["trilinear"], align_corners=True),
                Resized(keys=["image2d"], spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image3d", "image2d"], k=256, mode="constant", constant_values=0),
                
                ToTensord(keys=["image3d", "image2d"],),
            ]
        )

        self.train_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.train_image3d_files, self.train_image2d_files], 
            transform=self.train_transforms,
            length=1000,
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
                LoadImaged(keys=["image3d", "image2d"]),
                AddChanneld(keys=["image3d", "image2d"],),
                Spacingd(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"]),  
                Rotate90d(keys=["image2d"], k=3),
                OneOf([
                    Orientationd(keys=('image3d'), axcodes="ARI"),
                    Orientationd(keys=('image3d'), axcodes="PRI"),
                    Orientationd(keys=('image3d'), axcodes="ALI"),
                    Orientationd(keys=('image3d'), axcodes="PLI"),
                    Orientationd(keys=["image3d"], axcodes="LAI"),
                    Orientationd(keys=["image3d"], axcodes="RAI"),
                    Orientationd(keys=["image3d"], axcodes="LPI"),
                    Orientationd(keys=["image3d"], axcodes="RPI"),
                    ],
                ), 
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                #         a_min=-500, #-200, 
                #         a_max=3071, #1500,
                #         b_min=0.0,
                #         b_max=1.0),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                #         a_min=-200, 
                #         a_max=1500,
                #         b_min=0.0,
                #         b_max=1.0),
                ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                        a_min=-500, #-200, 
                        a_max=3071, #1500,
                        b_min=-500,
                        b_max=3071),
                # ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                #         a_min=-200, 
                #         a_max=1500,
                #         b_min=-200,
                #         b_max=1500),
                Lambdad(keys=["image3d"], func=HU2Density),
                ScaleIntensityd(keys=["image3d"], 
                        minv=0.0,
                        maxv=1.0),
                Resized(keys=["image3d"], spatial_size=256, size_mode="longest", mode=["trilinear"], align_corners=True),
                Resized(keys=["image2d"], spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image3d", "image2d"], k=256, mode="constant", constant_values=0),
            
                ToTensord(keys=["image3d", "image2d"],),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.val_image3d_files, self.val_image2d_files], 
            transform=self.val_transforms,
            length=200,
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

class OrDiffDataModule(LightningDataModule):
    def __init__(self, 
        train_image3d_folders: str = "path/to/folder", 
        train_image2d_folders: str = "path/to/folder", 
        val_image3d_folders: str = "path/to/folder", 
        val_image2d_folders: str = "path/to/folder", 
        test_image3d_folders: str = "path/to/folder", 
        test_image2d_folders: str = "path/to/dir", 
        train_samples: int = 1000,
        val_samples: int = 400,
        test_samples: int = 400,
        shape: int = 256,
        batch_size: int = 32
    ):
        super().__init__()

        self.batch_size = batch_size
        self.shape = shape
        # self.setup() 
        self.train_image3d_folders = train_image3d_folders
        self.train_image2d_folders = train_image2d_folders
        self.val_image3d_folders = val_image3d_folders
        self.val_image2d_folders = val_image2d_folders
        self.test_image3d_folders = test_image3d_folders
        self.test_image2d_folders = test_image2d_folders
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
            
        self.train_image3d_files = glob_files(folders=train_image3d_folders, extension='**/*.nii.gz')
        self.train_image2d_files = glob_files(folders=train_image2d_folders, extension='**/*.png')
        
        self.val_image3d_files = glob_files(folders=val_image3d_folders, extension='**/*.nii.gz') # TODO
        self.val_image2d_files = glob_files(folders=val_image2d_folders, extension='**/*.png')
        
        self.test_image3d_files = glob_files(folders=test_image3d_folders, extension='**/*.nii.gz') # TODO
        self.test_image2d_files = glob_files(folders=test_image2d_folders, extension='**/*.png')

    def setup(self, seed: int=2222, stage: Optional[str]=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image3d", "image2d"]),
                AddChanneld(keys=["image3d", "image2d"],),
                Spacingd(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"]),  
                Rotate90d(keys=["image2d"], k=3),
                OneOf([
                    Orientationd(keys=('image3d'), axcodes="ARI"),
                    # Orientationd(keys=('image3d'), axcodes="PRI"),
                    # Orientationd(keys=('image3d'), axcodes="ALI"),
                    # Orientationd(keys=('image3d'), axcodes="PLI"),
                    # Orientationd(keys=["image3d"], axcodes="LAI"),
                    # Orientationd(keys=["image3d"], axcodes="RAI"),
                    # Orientationd(keys=["image3d"], axcodes="LPI"),
                    # Orientationd(keys=["image3d"], axcodes="RPI"),
                    ],
                ),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                OneOf([
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                            a_min=-200, 
                            a_max=1500,
                            b_min=0.0,
                            b_max=1.0),
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                            a_min=-500, #-200, 
                            a_max=3071, #1500,
                            b_min=0.0,
                            b_max=1.0),
                ]),
                # RandZoomd(keys=["image3d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["trilinear"], align_corners=True), 
                # RandZoomd(keys=["image2d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["area"]), 
                RandFlipd(keys=["image2d"], prob=1.0, spatial_axis=1),
                # RandFlipd(keys=["image3d"], prob=0.5, spatial_axis=0),
                # RandFlipd(keys=["image3d"], prob=0.5, spatial_axis=1),

                RandScaleCropd(keys=["image3d"], 
                               roi_scale=(0.9, 0.9, 0.8), 
                               max_roi_scale=(1.0, 1.0, 0.8), 
                               random_center=False, 
                               random_size=False),
                # RandAffined(keys=["image3d"], rotate_range=None, shear_range=None, translate_range=20, scale_range=None),
                # CropForegroundd(keys=["image3d"], source_key="image3d", select_fn=lambda x: x>0, margin=0),
                # CropForegroundd(keys=["image2d"], source_key="image2d", select_fn=lambda x: x>0, margin=0),
                Resized(keys=["image3d"], spatial_size=self.shape, size_mode="longest", mode=["trilinear"], align_corners=True),
                Resized(keys=["image2d"], spatial_size=self.shape, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image3d", "image2d"], k=self.shape, mode="constant", constant_values=0),
                
                ToTensord(keys=["image3d", "image2d"],),
            ]
        )

        self.train_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.train_image3d_files, self.train_image2d_files], 
            transform=self.train_transforms,
            length=self.train_samples,
            batch_size=self.batch_size,
        )

        self.train_loader = DataLoader(
            self.train_datasets, 
            batch_size=self.batch_size, 
            num_workers=4, 
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image3d", "image2d"]),
                AddChanneld(keys=["image3d", "image2d"],),
                Spacingd(keys=["image3d"], pixdim=(1.0, 1.0, 1.0), mode=["bilinear"]),  
                Rotate90d(keys=["image2d"], k=3),
                RandFlipd(keys=["image2d"], prob=1.0, spatial_axis=1), #Right cardio
                OneOf([
                    Orientationd(keys=('image3d'), axcodes="ARI"),
                    # Orientationd(keys=('image3d'), axcodes="PRI"),
                    # Orientationd(keys=('image3d'), axcodes="ALI"),
                    # Orientationd(keys=('image3d'), axcodes="PLI"),
                    # Orientationd(keys=["image3d"], axcodes="LAI"),
                    # Orientationd(keys=["image3d"], axcodes="RAI"),
                    # Orientationd(keys=["image3d"], axcodes="LPI"),
                    # Orientationd(keys=["image3d"], axcodes="RPI"),
                    ],
                ), 
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0,),
                OneOf([
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # CTXR range
                            a_min=-200, 
                            a_max=1500,
                            b_min=0.0,
                            b_max=1.0),
                    ScaleIntensityRanged(keys=["image3d"], clip=True,  # Full range
                            a_min=-500, #-200, 
                            a_max=3071, #1500,
                            b_min=0.0,
                            b_max=1.0),
                ]),
                Resized(keys=["image3d"], spatial_size=self.shape, size_mode="longest", mode=["trilinear"], align_corners=True),
                Resized(keys=["image2d"], spatial_size=self.shape, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image3d", "image2d"], k=self.shape, mode="constant", constant_values=0),
            
                ToTensord(keys=["image3d", "image2d"],),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image3d", "image2d"],
            data=[self.val_image3d_files, self.val_image2d_files], 
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

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--shape", type=int, default=256, help="isotropic shape")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    hparams = parser.parse_args()
    # Create data module
    train_image3d_folders = [
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/train/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/val/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2019/raw/test/rawdata/'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/Verse2020/raw/test/rawdata/'),

        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/train/images'),
        # os.path.join(hparams.datadir, 'SpineXRVertSegmentation/UWSpine/processed/test/images/'),

        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/NSCLC/processed/train/images'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-0'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-1'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-2'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-3'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/MOSMED/processed/train/images/CT-4'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Imagenglab/processed/train/images'),
    ]
    train_label3d_folders = [

    ]

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

    val_image3d_folders = train_image3d_folders
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

    test_image3d_folders = val_image3d_folders
    test_image2d_folders = val_image2d_folders

    datamodule = OrDiffDataModule(
        train_image3d_folders = train_image3d_folders, 
        train_image2d_folders = train_image2d_folders, 
        val_image3d_folders = val_image3d_folders, 
        val_image2d_folders = val_image2d_folders, 
        test_image3d_folders = test_image3d_folders, 
        test_image2d_folders = test_image2d_folders, 
        batch_size = hparams.batch_size, 
        shape = hparams.shape
    )
    datamodule.setup(seed=hparams.seed)


