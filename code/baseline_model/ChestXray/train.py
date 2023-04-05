from tqdm import trange
from os.path import join

from dataloader import ChestXray

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision
from sklearn.metrics import roc_auc_score,roc_curve,precision_recall_fscore_support


def load_data(path, shuffle=False, seed=None, bs=1):
    train_dataset = ChestXray(
        dataset_path=join(path, "train"),
        data_shuffle=shuffle,
        seed=seed,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    test_dataset = ChestXray(
        dataset_path=join(path, "test"),
        data_shuffle=shuffle,
        seed=seed,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    return train_dataloader, test_dataloader

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]

def five_scores(labels, predictions):
    fpr, tpr, threshold = roc_curve(labels, predictions, pos_label=1)
    fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
    auc_value = roc_auc_score(labels, predictions)
    this_class_label = np.array(predictions)
    this_class_label[this_class_label>=threshold_optimal] = 1
    this_class_label[this_class_label<threshold_optimal] = 0
    predictions = this_class_label
    precision, recall, fscore, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    accuracy = 1- np.count_nonzero(np.array(labels).astype(int)- predictions.astype(int)) / len(labels)
    return accuracy, auc_value, precision, recall, fscore

def train(train_dataloader, epoch):
    model = torchvision.models.resnet50(weights="DEFAULT", progress=True)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, model.fc.out_features),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(model.fc.in_features, 3),
    )

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)

    score_dict = {}
    preds = []
    targets = []
    for i in trange(epoch):
        model.train()
        for target, input in train_dataloader:
            optimizer.zero_grad()
            pred = model(input)
            # print(type(target))
            # target = torch.tensor(target)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            preds.extend(pred)
            targets.extend(target)
        scheduler.step()
    score_dict['acc'], score_dict['auc'], score_dict['precision'], score_dict['recall'], score_dict['fscore'] = five_scores(targets,preds)
    torch.save(model.state_dict(),'state_dict.pt')
    pd.DataFrame.from_dict(score_dict).to_csv('score_dict.csv')
    return model


if __name__ == "__main__":
    train_dataloader, test_dataloader = load_data(
        path="/Volumes/T7 Shield/6211H_Final/data/Chest_X-ray_(Covid-19&Pneumonia)/",
        shuffle=True,
        seed=2023,
        bs=64,
    )
    model = train(train_dataloader=train_dataloader,epoch=10)
    print('fin')
