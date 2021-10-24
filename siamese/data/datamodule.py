import torch
import sklearn
import numpy as np
import pandas as pd

from tqdm import tqdm

import pytorch_lightning as pl

from PIL import Image, ImageOps

from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from siamese.transform.pair_transform import *
from sklearn.model_selection import train_test_split

from typing import *

def transform_fn(train=False):
    if train:                                 
        return PairCompose([
            PairResize((224,224)),
            PairRandomRotation(20),
            PairToTensor(),
        ])
    else:
        return PairCompose([
            PairResize((224,224)),
            PairToTensor(),
        ])

class GeneralDataset(Dataset):
    def __init__(self, root, train: bool = True, val_size: float = 0.2,
                 main_transform=None, pair_transform=None, comp_transform=None, 
                 invert=False, **kwargs):
        super(GeneralDataset).__init__()
        self.root = root
        self.train = train
        self.val_size = val_size
        self.main_transform = main_transform
        self.pair_transform = pair_transform
        self.comp_transform = comp_transform
        self.invert = invert
        self._dataset = ImageFolder(root=self.root)
        self.label_image_dict = self._create_label_image_dict()
        
        # initialize dataframe
        self._build_pairset_dataframe()
        self._split_train_valid_dataframe()
        
    
    @property
    def classes(self):
        return self._dataset.classes
    
    @property
    def class_to_idx(self):
        return self._dataset.class_to_idx

    def _build_pairset_dataframe(self):
        simm_df = self._create_similar_pair()
        diff_df = self._create_different_pair()
        diff_df = self._balance_different_pair(simm_df, diff_df)
        self.dataframe = self._combine_simm_diff_pair(simm_df, diff_df)
    
    def _split_train_valid_dataframe(self):
        label_name_list = self.dataframe['main_label_idx'].unique().tolist()
        df_simm = self.dataframe[self.dataframe['label']==0].reset_index(drop=True)
        df_diff = self.dataframe[self.dataframe['label']==1].reset_index(drop=True)

        simm_train_df, simm_valid_df = self._split_train_valid_percategory(df_simm, label_name_list, test_size=self.val_size)
        diff_train_df, diff_valid_df = self._split_train_valid_percategory(df_diff, label_name_list, test_size=self.val_size)

        train_df = pd.concat([simm_train_df, diff_train_df])
        train_df = sklearn.utils.shuffle(train_df, random_state=1261).reset_index(drop=True)

        valid_df = pd.concat([simm_valid_df, diff_valid_df])
        valid_df = sklearn.utils.shuffle(valid_df, random_state=1261).reset_index(drop=True)
        
        self.valid_dataframe = valid_df
        self.train_dataframe = train_df
        
        if self.train:
            self.dataframe = train_df
        else:
            self.dataframe = valid_df
    
    
    
    def _split_train_valid_percategory(self, df_data, label_name_list, test_size=0.2, random_state=1261):
        train_df_list, valid_df_list = [], []
        for idx in tqdm(range(len(label_name_list))):
            df_data_percat = df_data[df_data['main_label_idx']==label_name_list[idx]].reset_index(drop=True)
            train_df_percat, valid_df_percat = train_test_split(df_data_percat, test_size=test_size, random_state=random_state)
            train_df_list.append(train_df_percat)
            valid_df_list.append(valid_df_percat)
        train_df, valid_df = pd.concat(train_df_list), pd.concat(valid_df_list)
        return train_df, valid_df


    def _create_label_image_dict(self):
        img_label_list = sorted(self._dataset.imgs, key=lambda x: x[1],  reverse=False)
        label_image_dict = {lbl: [] for img, lbl in img_label_list}
        for impath, label in img_label_list:
            label_image_dict[label].append(impath)
        return label_image_dict
    

    def _create_similar_pair(self):
        # create pair same pair
        simm_pair = {'main_image': [], 'main_label_idx':[], 'main_label_name': [], 
                    'comp_image': [], 'comp_label_idx':[], 'comp_label_name': [], 
                    'label': [], 'status':[]}
        for key, list_value in self.label_image_dict.items():
            for idx, main_img in enumerate(list_value):
                for jdx, comp_image in enumerate(list_value):
                    if idx!=jdx:
                        simm_pair['main_image'].append(main_img)
                        simm_pair['main_label_name'].append(self._dataset.classes[key])
                        simm_pair['main_label_idx'].append(key)
                        simm_pair['comp_image'].append(comp_image)
                        simm_pair['comp_label_name'].append(self._dataset.classes[key])
                        simm_pair['comp_label_idx'].append(key)
                        simm_pair['label'] = int(key != key)
                        simm_pair['status'] = 'similar'
        simm_df = pd.DataFrame(simm_pair) 
        return simm_df

    def _create_different_pair(self):
        diff_pair = {'main_image': [], 'main_label_idx':[], 'main_label_name': [], 
             'comp_image': [], 'comp_label_idx':[], 'comp_label_name': [], 
             'label': [], 'status':[]}
        for main_key, main_list_value in tqdm(self.label_image_dict.items()):
            for diff_key, diff_list_value in self.label_image_dict.items():
                if main_key!=diff_key:
                    for idx, main_img in enumerate(main_list_value):
                        for jdx, comp_image in enumerate(diff_list_value):
                            diff_pair['main_image'].append(main_img)
                            diff_pair['main_label_name'].append(self._dataset.classes[main_key])
                            diff_pair['main_label_idx'].append(main_key)
                            diff_pair['comp_image'].append(comp_image)
                            diff_pair['comp_label_name'].append(self._dataset.classes[diff_key])
                            diff_pair['comp_label_idx'].append(diff_key)
                            diff_pair['label'] = int(main_key != diff_key)
                            diff_pair['status'] = 'different'
        diff_df = pd.DataFrame(diff_pair)
        return diff_df

    def _balance_different_pair(self, simm_df, diff_df, random_state=1261):
        diff_df_list = []
        for idx, name in enumerate(self._dataset.classes):
            label_name = self._dataset.classes[idx]
            simm_df_by_idx = simm_df[simm_df['main_label_name'] == label_name]
            len_simm_idx = len(simm_df_by_idx)

            label_name = self._dataset.classes[idx]
            diff_df_by_idx = diff_df[diff_df['main_label_name'] == label_name]
            len_diff_idx = len(diff_df_by_idx)

            balance_ratio = len_simm_idx / len_diff_idx
            diff_df_ratio_idx = diff_df_by_idx.sample(frac=balance_ratio, random_state=random_state).reset_index(drop=True)
            diff_df_list.append(diff_df_ratio_idx)

        diff_df = pd.concat(diff_df_list)
        return diff_df
    
    def _combine_simm_diff_pair(self, simm_df, diff_df, shuffle=True, random_state=1261):
        main_df = pd.concat([simm_df, diff_df])
        main_df = main_df.reset_index(drop=True)
        if shuffle:
            main_df = sklearn.utils.shuffle(main_df, random_state=1261)
            main_df = main_df.reset_index(drop=True)
        return main_df
    
    def _load_image(self, path: str, to_rgb=True):
        image = Image.open(path)
        if to_rgb:
            image = image.convert("RGB")
        else:
            image = image.convert("L")
        return image
    
    def _preprocess_label(self, label):
        label_numpy = np.array([label],dtype=np.float32)
        return torch.from_numpy(label_numpy)
                    
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        record = self.dataframe.iloc[idx]
        main_path, comp_path = record['main_image'], record['comp_image']
        main_image, comp_image = self._load_image(main_path), self._load_image(comp_path)
        label = self._preprocess_label(record['label'])
        
        if self.invert:
            main_image = ImageOps.invert(main_image)
            comp_image = ImageOps.invert(comp_image)


        if self.pair_transform:
            main_image, comp_image = self.pair_transform(main_image, comp_image)
        
        if self.main_transform:
            main_image = self.main_transform(main_image)
            
        if self.comp_transform:
            comp_image = self.comp_transform(comp_image)
        
        return main_image, comp_image, label
    
    def get(self, idx):
        record = self.dataframe.iloc[idx]
        main_path, main_label_name, main_label_idx = record['main_image'], record['main_label_name'], record['main_label_idx']
        comp_path, comp_label_name, comp_label_idx = record['comp_image'], record['comp_label_name'], record['comp_label_idx']
        
        
        main_image, comp_image = self._load_image(main_path), self._load_image(comp_path)
        label = self._preprocess_label(record['label'])
        
        if self.invert:
            main_image = ImageOps.invert(main_image)
            comp_image = ImageOps.invert(comp_image)


        if self.pair_transform:
            main_image, comp_image = self.pair_transform(main_image, comp_image)
        
        if self.main_transform:
            main_image = self.main_transform(main_image)
            
        if self.comp_transform:
            comp_image = self.comp_transform(comp_image)
        
        return (main_image, main_label_name, main_label_idx), (comp_image, comp_label_name, comp_label_idx), label

class OmniglotDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 128, num_workers: int = 8, **kwargs):
        super(OmniglotDataModule, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.train_transform = transform_fn(train=True)
        self.valid_transform = transform_fn(train=False)
        
    def setup(self, stage: Optional[str] = None):
        self.omniglot_trainset = GeneralDataset(root=self.data_dir, pair_transform=self.train_transform, train=True)
        self.omniglot_validset = GeneralDataset(root=self.data_dir, pair_transform=self.valid_transform, train=False)

    def train_dataloader(self):
        return DataLoader(self.omniglot_trainset, batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.omniglot_validset, batch_size=self.batch_size, num_workers=self.num_workers)
        