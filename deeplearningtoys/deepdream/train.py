'''
Function:
    模型训练
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import torch
import argparse
import numpy as np
import scipy.ndimage as ndimage
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from modules import getImagePyramid, randomShift, reconstructImage, matchFeatures, touchdir, DeepDream


'''命令行参数解析'''
def parseArgs():
    parser = argparse.ArgumentParser(description='让卷积神经网络做一个梦')
    parser.add_argument('--lr', dest='lr', help='初始学习率', type=float, required=False, default=2e-2)
    parser.add_argument('--iters', dest='iters', help='训练迭代次数', type=int, required=False, default=50)
    parser.add_argument('--backbone', dest='backbone', help='使用的骨干网络', default='resnet50', type=str, required=False)
    parser.add_argument('--savedir', dest='savedir', help='保存结果的路径', default='outputs', type=str, required=False)
    parser.add_argument('--imagepath', dest='imagepath', help='用来做梦的图片', type=str, required=True)
    parser.add_argument('--controlimagepath', dest='controlimagepath', help='用来控制神经网络梦境的图片, 不指定则不控制', default=None, type=str, required=False)
    parser.add_argument('--saveinterval', dest='saveinterval', help='模型保存的iter间隔', type=int, required=False, default=10)
    parser.add_argument('--jitter', dest='jitter', help='抖动参数', type=float, required=False, default=32)
    parser.add_argument('--octaves', dest='octaves', help='图像金字塔层数', type=float, required=False, default=6)
    parser.add_argument('--scale', dest='scale', help='图像金字塔scale', type=float, required=False, default=1.4)
    args = parser.parse_args()
    return args


'''训练器'''
class Trainer():
    def __init__(self, opts, **kwargs):
        self.opts = opts
    '''模型训练接口'''
    def train(self):
        # 基础准备
        means, stds = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        touchdir(self.opts.savedir)
        use_cuda = torch.cuda.is_available()
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        img_ori_trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize(means, stds)])
        img_control_trans = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(means, stds)])
        # 模型定义
        model = DeepDream(backbone_type=self.opts.backbone)
        if use_cuda: model = model.cuda()
        model.train()
        model.unlearnable()
        # 读取图像
        img_ori = Image.open(self.opts.imagepath)
        img_ori = img_ori_trans(img_ori).unsqueeze(0)
        features_control = None
        if self.opts.controlimagepath:
            img_control = img_control_trans(Image.open(self.opts.controlimagepath)).unsqueeze(0)
            with torch.no_grad(): features_control = model(img_control.type(FloatTensor))
        # 模型训练
        means, stds = np.array(means).reshape([3, 1, 1]), np.array(stds).reshape([3, 1, 1])
        img_pyramid = getImagePyramid(img_ori.numpy(), num_octaves=self.opts.octaves, octave_scale=self.opts.scale)
        diff = np.zeros_like(img_pyramid[-1])
        for idx, octave in enumerate(img_pyramid[::-1]):
            print('START OCTAVE: %s/%s' % (idx+1, len(img_pyramid)))
            if idx > 0:
                h, w = octave.shape[-2:]
                h_ori, w_ori = diff.shape[-2:]
                diff = ndimage.zoom(diff, (1, 1, 1.0*h/h_ori, 1.0*w/w_ori), order=1)
            octave_input = octave + diff
            for iteration in tqdm(range(1, self.opts.iters+1)):
                shift_x, shift_y = np.random.randint(-self.opts.jitter, self.opts.jitter+1, 2)
                octave_input = randomShift(octave_input, shift_x, shift_y)
                octave_input = torch.from_numpy(octave_input).type(FloatTensor)
                octave_input.requires_grad = True
                model.zero_grad()
                output = model(octave_input)
                matched = matchFeatures(output, features_control, FloatTensor)
                output.backward(matched)
                lr = self.opts.lr / np.abs(octave_input.grad.data.cpu().numpy()).mean()
                octave_input.data.add_(octave_input.grad.data * lr)
                if iteration % self.opts.saveinterval == 0:
                    savepath = os.path.join(self.opts.savedir, '%s_%s.jpg' % (idx, iteration))
                    img_reconstruct = reconstructImage(octave_input, shift_x, shift_y, means, stds, savepath)
                else:
                    img_reconstruct = reconstructImage(octave_input, shift_x, shift_y, means, stds, None, False)
                octave_input = img_reconstruct
            diff = img_reconstruct - octave


'''run'''
if __name__ == '__main__':
    args = parseArgs()
    client = Trainer(opts=args)
    client.train()