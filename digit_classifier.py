import torch
import torch.nn as nn
import torch.nn.functional as F

class Digit_Classifier(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Digit_Classifier, self).__init__()

        [kernel1, kernel2] = kernel_size
        [stride1, stride2] = stride

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=kernel1, stride=stride1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=kernel2, stride=stride2)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.out = nn.Linear(6272, 10)

    def forward(self, inputs):
        x = inputs.permute(0,3,1,2)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.out(x)
        return x
