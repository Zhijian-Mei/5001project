from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import random_split
from tqdm import tqdm

from data_util import *

import ssl

torch.manual_seed(48)
ssl._create_default_https_context = ssl._create_unverified_context


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
data_dir = "../data/Img_crop"
data_val = "../data/Eval"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 4

# Batch size for training (change depending on how much memory you have)
batch_size = 32

# Number of epochs to train for
num_epochs = 15

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



def initialize_model(model_name, num_classes,use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, True)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name, num_classes, use_pretrained=True)

# Print the model we just instantiated

dataset = get_dataset(image_size=input_size,data_path=data_dir,batch_size=batch_size)
dataset_val = get_dataset(image_size=input_size,data_path=data_val,batch_size=batch_size)

train_size = int(len(dataset))

train_dataset, eval_dataset = random_split(dataset, [train_size, 0])

trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
evalloader = DataLoader(dataset_val,batch_size=batch_size,shuffle=True)

model_ft.to(device)

optimizer = optim.AdamW(model_ft.parameters(), lr=0.0001)

criterion = nn.CrossEntropyLoss()
trainbar = tqdm(trainloader)
evalbar = tqdm(evalloader)

best_acc = -1
best_model = None

model_ft.train()
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(trainbar):
        images = images.to(device)
        labels = labels.to(device)
        out = model_ft(images)

        optimizer.zero_grad()
        loss = criterion(out,labels)
        loss.backward()
        optimizer.step()


    running_loss = 0.0
    running_corrects = 0
    model_ft.eval()
    for i, (images, labels) in enumerate(evalbar):
        images = images.to(device)
        labels = labels.to(device)
        out = model_ft(images)
        loss = criterion(out, labels)

        _, preds = torch.max(out, 1)

        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)

        epoch_acc = running_corrects.double() / len(dataset_val)
    print(f'epoch{epoch+1} accuracy: ',epoch_acc.item())
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model = copy.deepcopy(model_ft.state_dict())

torch.save(best_model,f'best_{model_name}_ft.pt')