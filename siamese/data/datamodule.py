
import pytorch_lightning as pl
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from torch.utils.data import Dataset, DataLoader
from transform.pair_transform import *

from typing import *

from .dataset import GeneralDataset
def transform_fn(train=False):
    if train:                                 
        return PairCompose([
            PairResize((112,112)),
            PairGrayscale(),
            PairToTensor(),
#             normalize
        ])
    else:
        return PairCompose([
            PairResize((112,112)),
            PairGrayscale(),
            PairToTensor(),
#             normalize
        ])


class OmniglotDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 128, num_workers: int = 8, simmilar_data_multiplier: int = 20, **kwargs):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.simmilar_data_multiplier = simmilar_data_multiplier
        self.train_transform = transform_fn(train=True)
        self.valid_transform = transform_fn(train=False)

        self._setup()

    def _setup(self, stage: Optional[str] = None):
        self.trainset = GeneralDataset(root=self.data_dir, simmilar_data_multiplier=self.simmilar_data_multiplier, pair_transform=self.train_transform, train=True)
        self.validset = GeneralDataset(root=self.data_dir, simmilar_data_multiplier=self.simmilar_data_multiplier, pair_transform=self.valid_transform, train=False)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)
        