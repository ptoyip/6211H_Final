from random import shuffle
import os
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose(
    [
        transforms.Grayscale(1),
        transforms.RandomRotation(30, fill=(0,)),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)


class CovidData(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg=None, state=None) -> None:
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg
        self.data_path = []
        self.labels = []  # [COVID,Lung_Opacity,Normal,Viral Pneumonia]
        dataset_path = os.path.join(self.dataset_cfg.data_dir, state)
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

        return img, label
