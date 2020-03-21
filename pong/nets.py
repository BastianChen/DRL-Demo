import torch
from torch import nn


# 封装卷积层
class CNNLayer(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, groups=1, bias=True):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, groups=groups, bias=bias),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU()
        )

    def forward(self, data):
        return self.layer(data)


# 封装残差层
class ResLayer(nn.Module):
    def __init__(self, input_channels):
        super().__init__()
        self.layer = nn.Sequential(
            CNNLayer(input_channels, input_channels // 2, 1, 1, 0),
            CNNLayer(input_channels // 2, input_channels // 2, 3, 1, 1),
            CNNLayer(input_channels // 2, input_channels, 1, 1, 0)
        )

    def forward(self, data):
        return data + self.layer(data)


# 主网络
class MainNet(nn.Module):
    def __init__(self):
        super().__init__()
        # 输入图片大小为[n,4,84,84]
        self.conv_layer = nn.Sequential(
            CNNLayer(4, 16, 3),  # n,16,82,82
            nn.MaxPool2d(3, 2),  # n,16,40,40
            ResLayer(16),
            ResLayer(16),
            CNNLayer(16, 32, 3),  # n,32,38,38
            nn.MaxPool2d(3, 2),  # n,32,18,18
            ResLayer(32),
            ResLayer(32),
            CNNLayer(32, 64, 3, 2, 1),  # n,64,9,9
            CNNLayer(64, 128, 3, 2, 1),  # n,128,5,5
        )
        self.linear_layer = nn.Sequential(
            nn.Linear(5 * 5 * 128, 64),
            nn.ReLU(),
        )
        # V值层
        self.value = nn.Linear(64, 1)
        # A优势函数层
        self.advantage = nn.Linear(64, 6)
        self.output_layer = nn.Linear(6, 6)
        self.mseloss = nn.MSELoss()

    def forward(self, data):
        data = self.conv_layer(data)
        data = data.reshape(data.size(0), -1)
        data = self.linear_layer(data)
        value = self.value(data)
        advantage = self.advantage(data)
        output = value + (advantage - advantage.mean())
        output = self.output_layer(output)
        return output

    def get_loss(self, writer, epoch, Q, Target_Q):
        loss = self.mseloss(Q, Target_Q)
        writer.add_scalar("loss", loss, epoch)
        return loss

    def add_histogram(self, writer, epoch):
        writer.add_histogram("weight/value", self.value.weight, epoch)
        writer.add_histogram("weight/advantage", self.advantage.weight, epoch)


if __name__ == '__main__':
    net = MainNet()
    params = sum([param.numel() for param in net.parameters()])
    print(params)
