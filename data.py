import os 
import glob

from typing import Callable, List, Optional, Sequence
from pytorch_lightning import LightningDataModule

from monai.data import Dataset, DataLoader
from monai.data import list_data_collate, decollate_batch
from monai.utils import set_determinism
from monai.transforms import (
    apply_transform, 
    Randomizable,
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

__all__ = [
    "UnpairedDataset", 
    "ImageDataModule", 
]

class UnpairedDataset(Dataset, Randomizable):
    def __init__(
        self, 
        keys: Sequence, 
        datasets: Sequence, 
        transform: Optional[Callable] = None,
        length: Optional[Callable] = None, 
        batch_size: int = 32
    ) -> None:
        self.keys = keys
        self.datasets = datasets
        self.length = length
        self.batch_size = batch_size
        self.transform = transform

    def __len__(self) -> int:
        if self.length is None:
            return min((len(dataset) for dataset in self.datasets))
        else: 
            return self.length

    def _transform(self, index: int):
        data = {}
        self.R.seed(index)
        for key, dataset in zip(self.keys, self.datasets):
            rand_idx = self.R.randint(0, len(dataset)) 
            data[key] = dataset[rand_idx]
        
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data

class ImageDataModule(LightningDataModule):
    def __init__(
        self,
        train_folders: List[str] = ["path/to/folder1", "path/to/folder2"],
        val_folders: List[str] = ["path/to/folder1", "path/to/folder2"], 
        test_folders: List[str] = ["path/to/folder1", "path/to/folder2"], 
        predict_folders: List[str] = ["path/to/folder1", "path/to/folder2"], 
        shape: int = 256, 
        batch_size: int = 32
    ) -> None:
            super().__init__()
            # self.train_folders = train_folders
            # self.val_folders = val_folders
            # self.test_folders = test_folders
            # self.predict_folders = predict_folders
            self.shape = shape
            self.batch_size = batch_size

            print(f"Dataset directories: ")
            def glob_files(folders: str=None, extension: str="*.nii.gz", verbose=True, k=1):
                print("="*80)
                print([folder for folder in folders])
                # assert folders is not None
                paths = [glob.glob(os.path.join(folder, extension), recursive = True) for folder in folders]
                files = sorted([item for sublist in paths for item in sublist])
                if verbose:
                    print(f"Total:", len(files))
                    print(f"First {k} file:", files[:k])
                return files
            self.train_files = glob_files(folders=train_folders, extension="**/*.png")
            self.val_files = glob_files(folders=val_folders, extension="**/*.png")
            self.test_files = glob_files(folders=test_folders, extension="**/*.png")
            self.predict_files = glob_files(folders=predict_folders, extension="**/*.png")

    def setup(self, seed: int=42, stage: Optional[str]=None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image2d"]),
                AddChanneld(keys=["image2d"]),
                Rotate90d(keys=["image2d"], k=3),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0),
                RandZoomd(keys=["image2d"], prob=1.0, min_zoom=0.9, max_zoom=1.0, padding_mode='constant', mode=["area"]), 
                RandFlipd(keys=["image2d"], prob=0.5, spatial_axis=1),
                Resized(keys=["image2d"], spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image2d"], k=256, mode="constant", constant_values=0),
                ToTensord(keys=["image2d"]),
            ]
        )

        self.train_datasets = UnpairedDataset(
            keys=["image2d"],
            data=[self.train_files, self.train_files], 
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
                LoadImaged(keys=["image2d"]),
                AddChanneld(keys=["image2d"]),
                Rotate90d(keys=["image2d"], k=3),
                ScaleIntensityd(keys=["image2d"], minv=0.0, maxv=1.0),
                Resized(keys=["image2d"], spatial_size=256, size_mode="longest", mode=["area"]),
                DivisiblePadd(keys=["image2d"], k=256, mode="constant", constant_values=0),
                ToTensord(keys=["image2d"]),
            ]
        )

        self.val_datasets = UnpairedDataset(
            keys=["image2d"],
            data=[self.val_files, self.val_files], 
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
