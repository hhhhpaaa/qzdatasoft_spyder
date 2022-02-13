import torch
from torch import nn


class CNNNet(nn.Module):

    def __init__(self):

        super(CNNNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=2, stride=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=36, kernel_size=2, stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(in_features=36*4*14, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=40)

    def forward(self, x):

        x = self.pool1(nn.functional.relu(self.conv1(x)))
        x = self.pool2(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 36*4*14)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        x = x.view(-1, 4, 10)

        return x


class DNNNet(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2, out_dim):

        super(DNNNet, self).__init__()
        self.layer1 = nn.Sequential(nn.Linear(in_dim, n_hidden_1),
                                    nn.BatchNorm1d(n_hidden_1),
                                    nn.Dropout(0.4))
        self.layer2 = nn.Sequential(nn.Linear(n_hidden_1, n_hidden_2),
                                    nn.BatchNorm1d(n_hidden_2),
                                    nn.Dropout(0.4))
        self.layer3 = nn.Sequential(nn.Linear(n_hidden_2, out_dim))

    def forward(self, x):
        x = x.view(-1, 22*62)
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = self.layer3(x)

        return x.view(-1, 4, 10)


if __name__ == '__main__':
    input = torch.randn(32, 1, 22, 62)
    # model = DNNNet(22*62, 1024, 512, 40)
    model = CNNNet()
    print(model(input).shape)
