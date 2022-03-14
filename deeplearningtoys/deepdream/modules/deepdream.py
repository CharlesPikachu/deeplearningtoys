'''
Function:
    定义DeepDream模型
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import torch
import torch.nn as nn
from .backbone import Backbone


'''定义DeepDream模型'''
class DeepDream(nn.Module):
    def __init__(self, backbone_type, **kwargs):
        super(DeepDream, self).__init__()
        self.classifier = Backbone(backbone_type=backbone_type, pretrained=True)
    '''forward'''
    def forward(self, x):
        return self.classifier(x)
    '''固定所有参数'''
    def unlearnable(self):
        for layer in self.modules():
            for param in layer.parameters():
                param.requires_grad = False