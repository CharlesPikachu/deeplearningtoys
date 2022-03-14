'''
Function:
    定义骨干网络
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import torch
import torchvision
import torch.nn as nn


'''定义骨干网络'''
class Backbone(nn.Module):
    def __init__(self, backbone_type, pretrained, **kwargs):
        super(Backbone, self).__init__()
        self.backbone_type = backbone_type.lower()
        self.backbone_used = getattr(torchvision.models, self.backbone_type)(pretrained=pretrained)
        if self.backbone_type in ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']:
            self.backbone_used = nn.Sequential(*[
                self.backbone_used.conv1, self.backbone_used.bn1, self.backbone_used.relu, self.backbone_used.maxpool, 
                self.backbone_used.layer1, self.backbone_used.layer2, self.backbone_used.layer3
            ])
    '''forward'''
    def forward(self, x):
        x = self.backbone_used(x)
        return x