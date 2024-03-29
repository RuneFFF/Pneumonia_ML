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
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin/'  # Adjust the path to your Graphviz installation directory

from sklearn.metrics import confusion_matrix
from seaborn import heatmap
from pandas import DataFrame
from torchview import draw_graph

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        # 1st convolutional layer, rectified linear unit function, dropout to reduce overfitting
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(0.3)

        # 2nd convolutional layer, rectified linear unit function
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.act2 = nn.ReLU()

        # 3rd convolutional layer, rectified linear unit function, max pooling of 3x3 squares for size reduction and lower translation sensitivity
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(3, 3), stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3))

        # flattening to prepare for linear layers (make 1D-array)
        self.flat = nn.Flatten()

        # linear layer with rectified linear unit function, dropout to reduce overfitting
        self.fc4 = nn.Linear(664000, 480)
        self.act4 = nn.ReLU()
        self.drop4 = nn.Dropout(0.5)

        # linear layer with rectified linear unit function
        self.fc5 = nn.Linear(480, 160)
        self.act5 = nn.ReLU()

        # last linear layer, 3 outputs for the 3 features "Normal", "Bacterial Pneumonia" and "Viral Pneumonia"
        self.fc6 = nn.Linear(160, 3)

    def forward(self, x):
        # input 1x500x750, output 16x500x750
        x = self.act1(self.conv1(x))
        x = self.drop1(x)
        # input 16x500x750, output 16x500x750
        x = self.act2(self.conv2(x))
        # input 16x500x750, output 16x500x750
        x = self.act3(self.conv3(x))
        # input 16x500x750, output 16x166x259
        x = self.pool3(x)
        # input 16x166x187, output 664000
        x = self.flat(x)
        # input 664000, output 480
        x = self.act4(self.fc4(x))
        x = self.drop4(x)
        # input 480, output 160
        x = self.act5(self.fc5(x))
        # input 160, output 3
        x = self.fc6(x)
        return x

# test loop
def do_test(epoch):
  net.eval()
  test_loss = []
  correct = 0
  all_outputs = []
  all_targets = []

  with torch.no_grad():
    for data, target in test_loader:
      all_targets.extend(target[:].cpu())
      if torch.cuda.is_available():
        data = data.to(torch.device("cuda:0"))
        target = target.type(torch.LongTensor).to(torch.device("cuda:0"))
      else: 
        target = target.type(torch.LongTensor)
      output = net(data)
      test_loss.append(criterion(output, target).item())
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
      all_outputs.extend(pred.T[0].cpu())

  test_loss_mean = np.mean(test_loss)
  test_losses.append(test_loss_mean)
  test_counter.append((epoch-1) * len(train_loader.dataset))
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss_mean, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  cf_matrix = confusion_matrix(all_targets, all_outputs, labels=[0, 1, 2])
  return cf_matrix

# train loop
def train(epoch):
    net.train()
    train_loss_epoch = []
    for batch_idx, (data, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            data = data.to(torch.device("cuda:0"))
            target = target.type(torch.LongTensor).to(torch.device("cuda:0"))  # cast target to tensor of type Long for Loss function
        opt.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        opt.step()
        train_loss_epoch.append(loss.item())
        train_losses.append(loss.item())
        train_counter.append((batch_idx * batch_size_train) + ((epoch - 2) * len(train_loader.dataset)))
    print('Train epoch: {} \tAvg. loss: {:.6f}'.format(
            epoch, np.mean(train_loss_epoch)))


if __name__ == '__main__':
    # hyperparameters
    n_epochs = 25
    batch_size_train = 64
    batch_size_test = 64
    learning_rate = 0.00005
    keep_training_with_best_model = True
    train_model = False

    # define model
    net = Network()
    if torch.cuda.is_available():
        # use GPU for network
        net.to(torch.device("cuda:0"))
        # enable ROCm/Cuda backend
        torch.backends.cudnn.enabled = True
    # define optimizer
    opt = optim.Adam(net.parameters(), lr=learning_rate)
    # define criterion
    criterion = F.cross_entropy

    # manual random seed for reproducibility
    random_seed = 1
    torch.manual_seed(random_seed)


    # read in csv which organizes data and provides labels
    data_set = XRaySet('chest_xray_data.csv', '../chest_xray', transform=transforms.Compose(
        [transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True), transforms.RandomHorizontalFlip(p=0.5)]))

    # split into training data (80%) and test data (20%)
    data_length = data_set.__len__()
    split1 = int(0.8 * data_length)
    split2 = int(0.2 * data_length)

    if split1 + split2 < data_length:  # fail save so whole set is split
        split2 = split2 + (data_length - split1 - split2)

    training_data, test_data = torch.utils.data.random_split(data_set, [split1, split2])

    # define dataloaders for testing and training with batching and shuffling
    train_loader = torch.utils.data.DataLoader(dataset=training_data, batch_size=batch_size_train, shuffle=True,
                                               pin_memory=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size_train, shuffle=True,
                                              pin_memory=True, num_workers=8)

    # counters for plotting
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = []

    # best losses / weights
    best_loss = inf
    best_net = net.state_dict()
    best_opt = opt.state_dict()

    # load model if saved state exists
    loaded_epoch = 0
    if os.path.exists(os.path.join(os.getcwd(), 'results', 'checkpoint.pth')):
        if torch.cuda.is_available():
            checkpoint = torch.load(os.path.join(os.getcwd(), 'results', 'checkpoint.pth'))
        else:
            checkpoint = torch.load(os.path.join(os.getcwd(), 'results', 'checkpoint.pth'),map_location=torch.device('cpu'))
        loaded_epoch = checkpoint['epoch']
        # if specified, use best available model
        if keep_training_with_best_model:
            net.load_state_dict(checkpoint['best_model_state_dict'])
            opt.load_state_dict(checkpoint['best_optimizer_state_dict'])
        # otherwise, use the most recent model
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

    if train_model: 
    # do_test(loaded_epoch)
        if loaded_epoch == 0:
            do_test(loaded_epoch)
        for epoch in range(1, n_epochs + 1):
            train(epoch+loaded_epoch)
            do_test(epoch+loaded_epoch)

            # update best model parameters if loss is better than previous best
            if test_losses[-1] < best_loss:
                best_loss = test_losses[-1]
                best_net = net.state_dict()
                best_opt = opt.state_dict()

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
    else:
        # feed test data and create confusion matrix
        cf_matrix = do_test(loaded_epoch)
        classes = ('Normal', 'Bacterial', 'Viral')
        df_cm = DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
        plt.figure(figsize = (12,7))
        heatmap(df_cm, annot=True)
        plt.savefig('confusion.svg')

        # plot network visualization
        model_graph = draw_graph(net,  input_size=(64,1,750,500), expand_nested=True)
        model_graph.visual_graph.render(format='svg')

    plt.rcParams.update({'font.size': 20})
    # plot evolution of loss in training and testing
    plt.figure('Loss Evolution')
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red', zorder=2)
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.ylim((0, 2))
    plt.grid()
    plt.xlabel('Number of Training Examples Seen')
    plt.ylabel('Cross Entropy Loss')

    # example data for plot with example images and assigned/true labels
    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    if torch.cuda.is_available():
        example_data.to(torch.device("cuda:0"))
        example_targets.to(torch.device("cuda:0"))

    with torch.no_grad():
        if torch.cuda.is_available():
            output = net(example_data.to(torch.device("cuda:0")))
        else:
            output = net(example_data)

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

    pass