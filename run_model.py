import torch.optim as optim
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from digit_classifier import Digit_Classifier

class RunModel():

    def __init__(self):
        self.loaded = False
        self.trained = False

    def load(self):
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize(0.5, 0.5)])

        trainset = torchvision.datasets.MNIST(root='./data', train=True,
                                              download=True, transform=transform)
        self.trainloader = DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.testloader = DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

    def train(self, epochs=100):
        criterion = nn.CrossEntropyLoss()
        model = Digit_Classifier()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        self.training_loss = []

        for epoch in range(epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    self.training_loss.append(running_loss / 2000)
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')
        self.trained = True
        self.model = model
        return model

    def test(self, model, dataloader):
        criterion = nn.CrossEntropyLoss()
        test_loss = []
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataloader:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                test_loss.append(criterion(outputs, labels.long()).item())

        self.avg_loss = np.mean(test_loss)
        self.accuracy = (100 * correct / total)
        return self.avg_loss, self.accuracy

    def show_metrics(self):
        if not self.trained:
            print('model has not been trained yet!')
            return

        print('Average')

    def show_trained_layers(self):
        if not self.trained:
            print('model has not been trained yet!')
            return


    def train_one(self, model, data_loader, optimizer, device=torch.device('cpu')):
        criterion = nn.CrossEntropyLoss()
        train_loss = []
        correct = 0
        total = 0

        for batch, labels in data_loader:
            batch, labels = batch.to(device), labels.to(device)
            optimizer.zero_grad()

            output = model(batch.float())
            loss = criterion(output, labels.long())
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            predicted = torch.max(output.data, 1)[1]

            for i in range(len(predicted)):
                correct += (1 if predicted[i] == labels[i] else 0)
                total += 1

        return model, np.mean(train_loss), (100 * correct / total)
