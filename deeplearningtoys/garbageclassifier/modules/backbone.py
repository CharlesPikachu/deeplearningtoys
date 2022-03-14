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
    def __init__(self, backbone_type, pretrained, num_classes, **kwargs):
        super(Backbone, self).__init__()
        self.backbone_type = backbone_type.lower()
        self.backbone_used = getattr(torchvision.models, self.backbone_type)(pretrained=pretrained)
        if self.backbone_type in ['alexnet', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn']:
            self.backbone_used.classifier[-1] = nn.Linear(in_features=4096, out_features=num_classes)
        elif self.backbone_type in ['inception_v3', 'resnet50', 'resnet101', 'resnet152']:
            self.backbone_used.fc = nn.Linear(in_features=2048, out_features=num_classes)
        elif self.backbone_type in ['resnet18', 'resnet34']:
            self.backbone_used.fc = nn.Linear(in_features=512, out_features=num_classes)
    '''forward'''
    def forward(self, x):
        x = self.backbone_used(x)
        return x