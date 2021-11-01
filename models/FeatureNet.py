
import __init__
from __init__ import *

import torch.nn as nn
import Meta
from Blocks import ResBlock

class FeatureNet(nn.Module):

    def __init__(self):
        super(FeatureNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(4, 8, kernel_size=(2,2), stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size = (2,2))
        self.batchnorm = nn.BatchNorm2d(8)
        self.fc1 = nn.Linear(16*8*8, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, Meta.data_meta['speakers'])

    def feature(self, x):
        x = x.view(-1, 1 ,256, 128)
        x = F.relu(self.maxpool(self.conv1(x)))
        x = F.relu(self.maxpool(self.conv2(x)))
        x = self.batchnorm(x)
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return x
        
    def forward(self, x):
        x = self.feature(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x
