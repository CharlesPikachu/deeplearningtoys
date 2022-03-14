'''
Function:
    定义一些工具函数
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cv2
import torch
import numpy as np
import scipy.ndimage as ndimage


'''图像金字塔'''
def getImagePyramid(img_ori, num_octaves=6, octave_scale=1.4):
    img_pyramid = [img_ori]
    for i in range(num_octaves-1):
        img_pyramid.append(ndimage.zoom(img_pyramid[-1], (1, 1, 1.0/octave_scale, 1.0/octave_scale), order=1))
    return img_pyramid


'''随机shift'''
def randomShift(data, shift_x, shift_y):
    return np.roll(np.roll(data, shift_x, -1), shift_y, -2)


'''图像重建'''
def reconstructImage(tensor, shift_x, shift_y, means, stds, savepath, is_save=True):
    img = tensor.data.cpu().numpy()
    img = np.roll(np.roll(img, -shift_x, -1), -shift_y, -2)
    img[0, ...] = np.clip(img[0, ...], -means/stds, (1-means)/stds)
    if is_save:
        means = means.reshape(1, 1, 3)
        stds = stds.reshape(1, 1, 3)
        img_save = img[0, ...].transpose(1, 2, 0)
        img_save = (stds * img_save + means) * 255
        img_save = np.uint8(np.clip(img_save, 0, 255))[..., (2, 1, 0)]
        cv2.imwrite(savepath, img_save)
    return img


'''特征匹配'''
def matchFeatures(output, features_control, FloatTensor):
    if features_control is None:
        return output.data
    output = output.data[0].cpu()
    c, h, w = output.size()
    output = output.view(output.size(0), -1)
    output = output.numpy().copy()
    features_control = features_control.data[0].cpu()
    features_control = features_control.view(features_control.size(0), -1)
    features_control = features_control.numpy().copy()
    dot_matrix = output.T.dot(features_control)
    return torch.from_numpy(features_control[:, dot_matrix.argmax(1)].reshape(c, h, w)).unsqueeze(0).type(FloatTensor)


'''新建文件夹'''
def touchdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        return False
    return True