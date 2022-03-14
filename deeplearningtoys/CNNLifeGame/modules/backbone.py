'''
Function:
    骨干网络
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import torch
import torch.nn as nn


'''ResNet的基础Block'''
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, with_shortcut=True, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels[0])
        self.conv2 = nn.Conv2d(in_channels[1], out_channels[1], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels[1])
        self.conv3 = nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels[2])
        if not with_shortcut:
            self.conv4 = nn.Conv2d(in_channels[0], out_channels[2], kernel_size=1, stride=1, padding=0, bias=False)
            self.bn4 = nn.BatchNorm2d(out_channels[2])
        self.relu = nn.ReLU(inplace=True)
        self.with_shortcut = with_shortcut
    '''forward'''
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if not self.with_shortcut:
            identity = self.conv4(identity)
            identity = self.bn4(identity)
        x += identity
        x = self.relu(x)
        return x


'''骨干网络'''
class Backbone(nn.Module):
    def __init__(self, **kwargs):
        super(Backbone, self).__init__()
        layer1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )
        layer2 = BasicBlock([8, 8, 8], [8, 8, 24], with_shortcut=False)
        layer3 = BasicBlock([24, 8, 8], [8, 8, 24], with_shortcut=True)
        layer4 = BasicBlock([24, 8, 8], [8, 8, 24], with_shortcut=True)
        layer5 = nn.Sequential(nn.Conv2d(24, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())
        self.layers = nn.ModuleList([layer1, layer2, layer3, layer4, layer5])
    '''forward'''
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x