import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import wandb
from hyper_params import *
import getpass
import streamlit as st
from rich.pretty import pprint

wandb.init('mnist_with_flax&jax')
wandb.config = {
    "learning_rate": learning_rate,
    "epochs": num_epochs,
    "batch_size": batch_size,
}

directory = f'{getpass.getuser()}/tai/project1/cnn/pytorch/'


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(64, 128, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            wandb.log({'training loss': loss, 'training_accuracy': 1 - loss})
            st.write('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

            pprint('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': 100. * correct / len(test_loader.dataset)
    })
    st.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    pprint('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    torch.manual_seed(125)
    st.info('Setting up device')
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': batch_size}
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    st.info('Processing dataset')
    dataset1 = datasets.MNIST(directory, train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST(directory, train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    st.info('Sending the model to device')
    model = ConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    scheduler = StepLR(optimizer, step_size=1, )
    progress_bar = st.progress(0)
    for index, epoch in enumerate(range(1, 100 + 1)):
        st.info('Training starting, the progress bar is initialized and will show the progress...')
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()
        progress_bar.progress(index)

    if True:
        torch.save(model.state_dict(), "mnist_cnn_pytorch.pt")
