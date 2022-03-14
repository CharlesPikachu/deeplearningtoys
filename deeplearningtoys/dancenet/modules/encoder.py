'''
Function:
    定义编码器
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''定义编码器'''
class Encoder(nn.Module):
    def __init__(self, image_size, **kwargs):
        super(Encoder, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        in_features = int(image_size[0] * image_size[1] * 32 // 64)
        self.fc = nn.Linear(in_features=in_features, out_features=512)
        self.fc_mean = nn.Linear(in_features=512, out_features=128)
        self.fc_logvar = nn.Linear(in_features=512, out_features=128)
    '''forward'''
    def forward(self, x):
        batch_size = x.size(0)
        x = self.convs(x)
        x = x.view(batch_size, -1)
        x = F.relu(self.fc(x), inplace=True)
        mean = self.fc_mean(x)
        logvar = self.fc_logvar(x)
        return mean, logvar