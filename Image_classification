#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data.dataset import Dataset
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.io import read_image
from PIL import Image
from glob import glob
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
import torch
import time
import csv
import os

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            # Stage 1
            nn.Conv2d(3, 3, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2),
            
            # Stage 2
            nn.Conv2d(3, 8, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2),

            # Stage 3
            nn.Conv2d(8, 16, kernel_size = 3, padding = 1),
            nn.Conv2d(16, 32, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2),
            
            # Stage 4
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2),

            # Stage 5
            nn.Conv2d(32, 32, kernel_size = 3, padding = 1),
            nn.Conv2d(32, 16, kernel_size = 3, padding = 1),
            nn.MaxPool2d(2),
        
        )
        self.fc = nn.Sequential(
            # fully-connected layer
            nn.Linear(1024, 1024),#4096 4096
            nn.ReLU(),
            nn.Linear(1024, 3),
            #nn.ReLU(),
            #nn.Linear(3, 1),
            #nn.ReLU()
        )


    def forward(self, x):
        x = self.conv(x)
        SizeforFC = x.size()[1]*x.size()[2]*x.size()[3] 
        x = x.view(-1, SizeforFC) #4096
        x = self.fc(x)
        return x.double()
        
def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(1):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            yhat = net.forward(images)
            loss = criterion(yhat, labels)
            # backward pass
            loss.backward()
            # optimize the network
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            #if i % 100 == 99:    # print every 2000 mini-batches
            end = time.time()
            print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                 (epoch + 1, i + 1, running_loss , end-start))
            start = time.time()
            running_loss = 0.0
    print('Finished Training')


def valid(validloader, net, device):
    correct = 0
    total = 0

    with torch.no_grad():
        for data in validloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            labels = np.array(labels)
            labels = np.array(np.argmax(labels,axis=1))
            outputs = np.array(net(images))
            class_id = np.array(np.argmax(outputs, axis=1))
    
            print(class_id)

            total += len(labels)
            correct += (class_id == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

def test(testSet, net, device, transform):

  name = 'drive/MyDrive/test_labels.csv'
  with open(name, 'w') as f:
      writer = csv.writer(f, delimiter=',', lineterminator='\n')
      writer.writerow(['guid/image', 'label'])
      with torch.no_grad():
        for data in testSet:
            images, imagepath = data
            imagepath = str(imagepath[0])
            guid = imagepath.split('/')[-2]
            idx = imagepath.split('/')[-1].replace('_image.jpg', '')
            images = images.to(device)
            outputs = net(images)
            class_id = np.array(np.argmax(outputs, axis=1))
            writer.writerow(['{}/{}'.format(guid, idx), class_id[0]])

  print('Wrote report file `{}`'.format(name))

class carData_test(Dataset):
    def __init__(self, imagePath, transform):
        
        self.imagePath = imagePath
        self.files = glob(self.imagePath)
        self.files.sort()
        self.tranform = transform
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        
        label = self.files[idx]
        image = Image.open(label).convert('RGB')
        image = self.tranform(image)
        return image, label

class carData(Dataset):
    def __init__(self, imagePath, csvFile, transform):
        self.imagePath = imagePath
        self.imageCSV = pd.read_csv(csvFile)
        self.tranform = transform
        self.label = np.array([0.0,0.0,0.0])
        
    def __len__(self):
        return len(self.imageCSV)
    
    def __getitem__(self, idx):
        imagePath = os.path.join(self.imagePath, self.imageCSV.iloc[idx, 0])
        image = Image.open(imagePath).convert('RGB')
        image = self.tranform(image)
        label = np.array([0.0,0.0,0.0])
        class_id = self.imageCSV.iloc[idx, 1]
        label[class_id] = 1.0;
        #print(label)
        return image, label


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomPerspective(),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

# Seperate training set and validation set
trainSetAll = carData('drive/MyDrive/rob535-fall2021-final-project-data/trainval/', 'drive/MyDrive/trainval_labels.csv', transform)
nTrain = round(len(trainSetAll) / 10 * 9)
nValid = round(len(trainSetAll) - len(trainSetAll) / 10 * 9)
trainSet, validSet = random_split(trainSetAll, [nTrain, nValid])

testSet = carData_test('drive/MyDrive/rob535-fall2021-final-project-data/test/*/*_image.jpg',transform)

trainLoader = DataLoader(trainSet, batch_size = 75, shuffle = True)
validLoader = DataLoader(validSet, batch_size = 50, shuffle = True)
testLoader = DataLoader(testSet, batch_size = 1, shuffle = False)
# Take validation set
# Typically, we should take all of the rest for validation, here we just want to make things work fast
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = VGG().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

train(trainLoader, net, criterion, optimizer, device)
valid(validLoader, net, device)
test(testLoader, net, device, transform)
