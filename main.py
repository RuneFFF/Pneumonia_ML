#test
#branch lorenz

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
#import torchvision
#import torchvision.transforms as transforms
#import numpy as np

#In case of 3-channel images
#class Net(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 5 * 5, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 10)
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = torch.flatten(x, 1) # flatten all dimensions except batch
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x

if __name__ == '__main__':
  
  net = nn.Module()
  
  #define criterion (loss) and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  
  #training loop
