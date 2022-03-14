'''
Function:
	定义VAE模型
Author:
	Charles
微信公众号:
	Charles的皮卡丘
'''
import torch
import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder


'''定义VAE模型'''
class DanceNet(nn.Module):
	def __init__(self, image_size, **kwargs):
		super(DanceNet, self).__init__()
		self.encoder = Encoder(image_size)
		self.decoder = Decoder(image_size)
		self.initweights()
	'''forward'''
	def forward(self, x):
		mean, logvar = self.encoder(x)
		std = logvar.mul(0.5).exp_()
		eps = torch.FloatTensor(std.size()).type_as(x).normal_()
		z = eps.mul(std).add_(mean)
		img = self.decoder(z)
		return img, mean, logvar
	'''权重初始化'''
	def initweights(self):
		def init(m):
			classname = m.__class__.__name__
			if classname.find('Conv2d') != -1:
				m.weight.data.normal_(0.0, 0.02)
				m.bias.data.fill_(0)
			elif classname.find('Linear') != -1:
				m.weight.data.normal_(0.0, 0.02)
				m.bias.data.fill_(0)
		self.encoder.apply(init)
		self.decoder.apply(init)