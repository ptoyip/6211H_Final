import random
import torch
import os
import pandas as pd
from pathlib import Path

import torch.utils.data as data
from torch.utils.data import dataloader


class CamelData(data.Dataset):
    def __init__(self, dataset_cfg=None, state=None):
        # Set all input args as attributes
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg

        # ---->data and label
        self.nfolds = self.dataset_cfg.nfold
        self.fold = self.dataset_cfg.fold
        self.feature_dir = self.dataset_cfg.data_dir
        self.csv_dir = self.dataset_cfg.label_dir + f"fold{self.fold}.csv"
        self.slide_data = pd.read_csv(self.csv_dir, index_col=0)

        # ---->order
        self.shuffle = self.dataset_cfg.data_shuffle

        # ---->split dataset
        if state == "train":
            self.data = self.slide_data.loc[:, "train"].dropna()
            self.label = self.slide_data.loc[:, "train_label"].dropna()
        if state == "val":
            self.data = self.slide_data.loc[:, "val"].dropna()
            self.label = self.slide_data.loc[:, "val_label"].dropna()
        if state == "test":
            self.data = self.slide_data.loc[:, "test"].dropna()
            self.label = self.slide_data.loc[:, "test_label"].dropna()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # print(idx, "is the current idx")
        slide_id = self.data[idx]
        slide_id = str(int(slide_id))
        label = int(self.label[idx])
        # print('\nnow print the slide id',str(slide_id))
        # full_path = Path(self.feature_dir) / f"slide{slide_id}.pt"
        full_path = Path(self.feature_dir) / f"{slide_id}.pt"
        # print('now print the full path: ',str(full_path))
        features = torch.load(full_path)

        # ----> shuffle
        if self.shuffle == True:
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        return features, label
    

from random import shuffle
from torchvision import transforms
from PIL import Image

transform = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.RandomRotation(30,fill=(0,)),
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(0.5,0.25)
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class ChestXray(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg=None, state=None) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_path = []
        self.labels = [] # ['COVID19', 'NORMAL', 'PNEUMONIA']
        dataset_path = os.path.join(self.dataset_cfg.data_dir,state)
        for folder in sorted(Path(dataset_path).iterdir()):
            self.labels.append(folder.stem)
            for file in folder.iterdir():
                file = str(file.resolve())
                self.data_path.append([folder.stem, file])

        if self.dataset_cfg.data_shuffle:
            shuffle(self.data_path)

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, index):
        assert index < self.__len__()

        label, path = self.data_path[index]
        img = transform(Image.open(path))
        label = self.labels.index(label)
        return label, img

