import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from create_XR_dataset import XRaySet
from matplotlib import pyplot as plt
import numpy as np
import timm

n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

#training_data = [data_set.__getitem__(i) for i in range(0, int(0.8*data_set.__len__()-1))]
#test_data = [data_set.__getitem__(i) for i in range(int(0.8*data_set.__len__()), data_set.__len__()-1)]

data_set = XRaySet('chest_xray_data.csv', '../chest_xray', transform=transforms.ToTensor())
data_length = data_set.__len__()
split1 = int(0.8*data_length)
split2 = int(0.2*data_length)+1


training_data, test_data = torch.utils.data.random_split(data_set, [split1, split2])

train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size_train, shuffle=True)

#### MNIST Data Test
# train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./files/',
#                             train=True, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))])),
#                             batch_size=batch_size_train, shuffle=True)
#
# test_loader = torch.utils.data.DataLoader(
#   torchvision.datasets.MNIST('./files/', train=False, download=True,
#                              transform=torchvision.transforms.Compose([
#                                torchvision.transforms.ToTensor(),
#                                torchvision.transforms.Normalize(
#                                  (0.1307,), (0.3081,))
#                              ])),
#   batch_size=batch_size_test, shuffle=True)


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)), #(1 Channel, 32 filters of shape (3,3)kernels
            nn.ReLU(),    #activation
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),
            nn.Flatten(), #some shaving of of 2 pixels per layer, therefore compensation is needed
            nn.Linear(64*(500-6)*(750-6),3)#output_size*dim1_img*dim2_img,number_outputclasses (ergo die 0,1,2 der label)
        )
    def forward(self, x):
            return self.model(x)

#################################
# class Network(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = torch.flatten(x, 1) # flatten all dimensions except batch
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#############################
# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
#         self.conv2_drop = nn.Dropout2d()
#         self.fc1 = nn.Linear(320, 50)
#         self.fc2 = nn.Linear(50, 10)
#
#     def forward(self, x):
#         x = F.relu(F.max_pool2d(self.conv1(x), 2))
#         x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
#         x = x.view(-1, 320)
#         x = F.relu(self.fc1(x))
#         x = F.dropout(x, training=self.training)
#         x = self.fc2(x)
#         return F.log_softmax(x)
########################

train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

def test():
  net.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = net(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        opt.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        opt.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
            torch.save(net.state_dict(), './results/model.pth')
            torch.save(opt.state_dict(), './results/optimizer.pth')


if __name__=='__main__':

    #define model
    net = Network()
    #timm.create_model('efficientnet_b4', pretrained=True, num_classes=2, in_chans=3)
    #define optimizer
    opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    #define criterion
    #criterion = F.cross_entropy
    criterion = F.nll_loss

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2, 3, i + 1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Type: {}".format(example_targets[i]))
    #     plt.xticks([])
    #     plt.yticks([])
    # plt.show()

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    with torch.no_grad():
        output = net(example_data)

    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    plt.show()



    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()

    # #loop
    # for i, (data, target) in enumerate(train_loader):
    #
    #     #set gradient to zero
    #     opt.zero_grad()
    #
    #
    #
    #     output = net(data)
    #     loss = criterion(output, target)
    #     loss.backward()
    #     opt.step()
    #
    #
    #     if i % log_interval == 0:
    #         print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #             epoch, batch_idx * len(data), len(train_loader.dataset),
    #                    100. * batch_idx / len(train_loader), loss.item()))
    #         train_losses.append(loss.item())
    #         train_counter.append(
    #             (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
    #         torch.save(net.state_dict(), './results/model.pth')
    #         torch.save(opt.state_dict(), './results/optimizer.pth')
    #
    #     if i % 10 == 0:
    #        print(f'[{i + 1:5d}] loss: {loss / 2000:.3f}')