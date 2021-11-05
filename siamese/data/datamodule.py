
import pytorch_lightning as pl


from torch.utils.data import Dataset, DataLoader
from siamese.transform.pair_transform import *

from typing import *

from .dataset import GeneralDataset
def transform_fn(train=False):
    if train:                                 
        return PairCompose([
            PairResize((80,80)),
            PairGrayscale(),
            PairToTensor(),
#             normalize
        ])
    else:
        return PairCompose([
            PairResize((80,80)),
            PairGrayscale(),
            PairToTensor(),
#             normalize
        ])


class OmniglotDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 128, num_workers: int = 8, **kwargs):
        super(OmniglotDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform = transform_fn(train=True)
        self.valid_transform = transform_fn(train=False)
        
    def setup(self, stage: Optional[str] = None):
        self.trainset = GeneralDataset(root=self.data_dir, pair_transform=self.train_transform, train=True)
        self.validset = GeneralDataset(root=self.data_dir, pair_transform=self.valid_transform, train=False)

    def train_dataloader(self):
        return DataLoader(self.trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.validset, batch_size=self.batch_size, num_workers=self.num_workers)
        