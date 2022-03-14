'''
Function:
    定义解码器
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


'''定义解码器'''
class Decoder(nn.Module):
    def __init__(self, image_size, **kwargs):
        super(Decoder, self).__init__()
        self.image_size = image_size
        self.fc1 = nn.Linear(in_features=128, out_features=512)
        out_features = int(image_size[0] * image_size[1] * 32 // 64)
        self.fc2 = nn.Linear(in_features=512, out_features=out_features)
        self.convs = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
    '''forward'''
    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = x.view(x.size(0), 32, self.image_size[1]//8, self.image_size[0]//8)
        x = self.convs(x)
        return x