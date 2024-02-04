import torch
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils
from create_XR_dataset import XRaySet
from matplotlib import pyplot as plt
import numpy as np
from math import inf
import os
from copy import deepcopy

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st convolutional layer, rectified linear unit function, dropout to reduce overfitting
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        # 2nd convolutional layer, rectified linear unit function, max pooling of 3x3 squares for size reduction and lower translation sensitivity
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3))

        # flattening to prepare for linear layers (make 1D-array)
        self.flat = nn.Flatten()

        # linear layer with rectified linear unit function, dropout to reduce overfitting
        self.fc3 = nn.Linear(664000, 480)
        self.act3 = nn.ReLU()
        self.drop3 = nn.Dropout(0.5)

        # last linear layer, 3 outputs for the 3 features "Normal", "Bacterial Pneumonia" and "Viral Pneumonia"
        self.fc4 = nn.Linear(480, 3)

    def forward(self, x):
        # input 1x500x750, output 32x500x750
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 32x500x750, output 32x500x750
        x = self.act2(self.conv2(x))
        # input 32x500x750, output 32x166x259
        x = self.pool2(x)
        # input 32x166x250, output 664000
        x = self.flat(x)
        # input 664000, output 480
        x = self.act3(self.fc3(x))
        x = self.drop3(x)
        # input 480, output 3
        x = self.fc4(x)
        return x

# test loop
def do_test():
  net.eval()
  test_loss = []
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data = data.to(torch.device("cuda:0"))
      output = net(data)
      target = target.type(torch.LongTensor).to(torch.device("cuda:0"))
      test_loss.append(criterion(output, target).item())
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss_mean = np.mean(test_loss)
  test_losses.append(test_loss_mean)
  test_counter.append(test_counter[-1] + len(train_loader.dataset))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss_mean, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))

# train loop
def train(epoch):
    net.train()
    train_loss_epoch = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(torch.device("cuda:0"))
        opt.zero_grad()
        output = net(data)
        target = target.type(torch.LongTensor).to(torch.device("cuda:0"))  # cast target to tensor of type Long for Loss function
        loss = criterion(output, target)
        loss.backward()
        opt.step()
        train_loss_epoch.append(loss.item())
        train_losses.append(loss.item())
        train_counter.append((batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))
    print('Train epoch: {} \tAvg. loss: {:.6f}'.format(
            epoch, np.mean(train_loss_epoch)))
    #save weights after each epoch (subject to change?)
    # torch.save(net.state_dict(), './results/model.pth')
    # torch.save(opt.state_dict(), './results/optimizer.pth')


if __name__=='__main__':
    # hyperparameters
    n_epochs = 3
    batch_size_train = 32
    batch_size_test = 32
    learning_rate = 0.001
    # momentum = 0.5
    keep_training_with_best_model = 0

    # manual random seed for reproducability
    random_seed = 1
    torch.manual_seed(random_seed)

    # enable ROCm/Cuda backend
    torch.backends.cudnn.enabled = True

    # read in csv which organzizes data and provides labels
    data_set = XRaySet('chest_xray_data.csv', '../chest_xray', transform=transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.RandomHorizontalFlip(p=0.5)]))

    # split into training data (80%) and test data (20%)
    data_length = data_set.__len__()
    split1 = int(0.8 * data_length)
    split2 = int(0.2 * data_length)

    if split1 + split2 < data_length:  # failsave so whole set is split
        split2 = split2 + (data_length - split1 - split2)

    training_data, test_data = torch.utils.data.random_split(data_set, [split1, split2])

    # define dataloaders for testing and training with batching and shuffling
    train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size_train, shuffle=True,
                                               pin_memory=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size_train, shuffle=True,
                                              pin_memory=True, num_workers=8)

    # define model
    net = Network()
    # use GPU for network
    net.to(torch.device("cuda:0"))
    # define optimizer
    #opt = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    opt = optim.Adam(net.parameters(), lr=learning_rate)
    # define criterion
    criterion = F.cross_entropy
    # criterion = F.nll_loss

    # counters for plotting
    train_losses = []
    train_counter = []
    test_losses = []
    #test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]
    test_counter = []

    # best losses / weights
    best_loss = inf
    best_net = net.state_dict()
    best_opt = opt.state_dict()

    # load model if saved state exists
    if os.path.exists(os.path.join(os.getcwd(),'results','checkpoint.pth')): 
        checkpoint = torch.load(os.path.join(os.getcwd(),'results','checkpoint.pth'))
        epoch = checkpoint['epoch'] # is this wanted?
        if keep_training_with_best_model:
            net.load_state_dict(checkpoint['best_model_state_dict'])
            opt.load_state_dict(checkpoint['best_optimizer_state_dict'])
        else:
            net.load_state_dict(checkpoint['current_model_state_dict'])
            opt.load_state_dict(checkpoint['current_optimizer_state_dict'])
        best_net = checkpoint['best_model_state_dict']
        best_opt = checkpoint['best_optimizer_state_dict']
        best_loss = checkpoint['loss']  
        train_losses = checkpoint['train_losses']
        train_counter = checkpoint['train_counter']
        test_losses = checkpoint['test_losses']
        test_counter = checkpoint['test_counter']

    do_test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        do_test()

        # update best model parameters if loss is better than previous best
        if test_losses[-1] < best_loss:
            best_loss = deepcopy(test_losses[-1])
            best_net = deepcopy(net.state_dict())
            best_opt = deepcopy(opt.state_dict())

        # save state of current and best model and optimizer, epoch and loss
        torch.save({
            'epoch': epoch, 
            'current_model_state_dict': net.state_dict(),
            'current_optimizer_state_dict': opt.state_dict(), 
            'best_model_state_dict': best_net,
            'best_optimizer_state_dict': best_opt, 
            'loss': test_losses[-1], 
            'train_losses': train_losses, 
            'train_counter': train_counter, 
            'test_losses': test_losses, 
            'test_counter': test_counter
        }, './results/checkpoint.pth') 


    # plot evolution of loss in training and testing
    plt.figure('Loss Evolution')
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.ylim((0, 2))
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Cross Entropy Loss')

    # example data for plot with example images and assigned/true labels
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    example_data.to(torch.device("cuda:0"))
    example_targets.to(torch.device("cuda:0"))

    with torch.no_grad():
        output = net(example_data.to(torch.device("cuda:0")))

    # plot examples
    plt.figure('Examples')
    for i in range(12):
        plt.subplot(3, 4, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        pred = output.data.max(1, keepdim=True)[1][i].item()
        if pred == 0:
            pred = 'Normal'
        elif pred == 1:
            pred = 'Bacterial'
        else:
            pred = 'Viral'
        label = example_targets[i]
        if label == 0:
            label = 'Normal'
        elif label == 1:
            label = 'Bacterial'
        else:
            label = 'Viral'
        plt.title("Prediction: "+pred+',\n Label: '+label)
    plt.show()