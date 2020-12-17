import torch
import torch.nn as nn
import torch.nn.functional as F

class Digit_Classifier(nn.Module):
    def __init__(self):
        super(Digit_Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6,16,3)
        self.pool2 = nn.MaxPool2d(2)
        self.out = nn.Linear(400, 10)

    def forward(self, inputs):
        x = self.pool1(F.relu(self.conv1(inputs)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        return x
