import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from create_XR_dataset import XRaySet

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

data_set = XRaySet('chest_xray_data.csv', '../chest_xray')

training_data = [data_set.__getitem__(i) for i in range(0, int(0.8*data_set.__len__()))]
test_data = [data_set.__getitem__(i) for i in range(int(0.8*data_set.__len__()), data_set.__len__())]

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size_train, shuffle=True)

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

if __name__=='__main__':
    net = Network()
    opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)