import os 
from argparse import ArgumentParser

from data import ImageDataModule

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape", type=int, default=256, help="isotropic shape")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    hparams = parser.parse_args()

    # Create data module
    train_folders = [
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/JSRT/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/ChinaSet/processed/images/'), 
        # os.path.join(hparams.datadir, 'ChestXRLungSegmentation/Montgomery/processed/images/'),
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/train/images/'), 
    ]

    val_folders = [
        os.path.join(hparams.datadir, 'ChestXRLungSegmentation/VinDr/v1/processed/test/images/'), 
    ]

    test_folders = val_folders
    predict_folders = val_folders

    datamodule = ImageDataModule(
        train_folders = train_folders, 
        val_folders = val_folders, 
        test_folders = test_folders, 
        predict_folders = predict_folders, 
        batch_size = hparams.batch_size, 
        shape = hparams.shape
    )
    
    datamodule.setup(seed=hparams.seed)