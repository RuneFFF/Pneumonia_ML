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

classes = ('normal', 'p_v', 'p_b')

if __name__ == '__main__':
    # for root,_,files in os.walk('chest_xray'):
    #     for filename in files:
    #         if filename[-5:] == ('.jpeg'):
    #             downsample(os.path.join(root,filename), 750, 500,os.path.join(root,filename))

    #x_rename('chest_xray')
    #makeCSV('chest_xray')
  
  batch_size = 4
  end_data_test = 624

  test_df = pd.read_csv('chest_xray_data.csv', nrows=end_data_test, header=0)

  test_data = XRaySet(test_df, root_dir = 'chest_xray/test', is_train=False)
  test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=2, shuffle=False)


  
  net = nn.Module()  #base class for neural networks
  
  #define criterion (loss) and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
  
  #training loop
for
