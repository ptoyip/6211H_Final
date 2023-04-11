from tqdm import trange
from os.path import join

from dataloader import ChestXray

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torchvision
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report


def load_data(path, shuffle=False, seed=None, bs=1):
    train_dataset = ChestXray(
        dataset_path=join(path, "train"),
        data_shuffle=shuffle,
        seed=seed,
    )
    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=8)
    splits = KFold(n_splits=10, shuffle=True, random_state=seed)
    test_dataset = ChestXray(
        dataset_path=join(path, "test"),
        data_shuffle=shuffle,
        seed=seed,
    )
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=8)
    return train_dataloader, test_dataloader


# def test(val_dataloader)

def train_one_epoch(model, device, criterion, optimizer,dataloader):
    train_loss,train_correct=0.0,0
    model.train()
    for data, labels in dataloader:
        data,labels = data.to(device),labels.to(device)
        optimizer.zero_grad()
        preds = model(data)
        loss = criterion(preds,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
        scores, predictions = torch.max(preds.data,dim = 1)
        train_correct += (predictions == labels).sum().item()
    return train_loss,train_correct

def valid_epoch(model,device,dataloader,loss_fn):
    valid_loss, val_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data,labels = data.to(device),labels.to(device)
            preds = model(data)
            loss=loss_fn(preds,labels)
            valid_loss+=loss.item()*data.size(0)
            scores, predictions = torch.max(preds.data,dim = 1)
            val_correct+=(predictions == labels).sum().item()

    return valid_loss,val_correct

def train(model, device, criterion, optimizer, scheduler,train_dataloader, epoch):
    for i in trange(epoch):
        model.train()
        for label, input in train_dataloader:
            optimizer.zero_grad()
            pred = model(input.to(device))
            # print(type(label))
            # label = torch.tensor(label)
            loss = criterion(pred, label.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
    torch.save(model.state_dict(), "state_dict.pt")
    return model


def model_init():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torchvision.models.resnet50(weights="DEFAULT", progress=True)
    model.conv1 = torch.nn.Conv2d(
        1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
    )
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.5),
        torch.nn.Linear(2048, 1000),
        torch.nn.Dropout(0.5),
        torch.nn.Linear(1000, 3),
        torch.nn.Softmax(dim=1),
    )
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5)
    return model, device, criterion, optimizer, scheduler


if __name__ == "__main__":
    train_dataloader, test_dataloader = load_data(
        # path="../../../data/Chest_X-ray_(Covid-19&Pneumonia)/",
        path="/jhcnas2/home/yipkc/ChestXray/",
        shuffle=True,
        seed=2023,
        bs=64,
    )
    model, device, criterion, optimizer, scheduler = model_init()
    model = train(model, device, criterion, optimizer, scheduler,train_dataloader=train_dataloader, epoch=100)
    print("fin")
