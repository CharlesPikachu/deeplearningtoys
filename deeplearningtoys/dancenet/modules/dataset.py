'''
Function:
    导入数据集
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


'''导入数据集'''
class ImageFolder(Dataset):
    def __init__(self, rootdir, image_size, **kwargs):
        self.image_size = image_size
        self.imagepaths = [os.path.join(rootdir, str(i)+'.jpg') for i in range(1, len(os.listdir(rootdir))+1)]
    '''__getitem__'''
    def __getitem__(self, index):
        img = cv2.imread(self.imagepaths[index], cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, self.image_size)
        img = img.astype(np.float32) / 255
        img = torch.from_numpy(img).unsqueeze(-1).permute(2, 0, 1)
        return img
    '''__len__'''
    def __len__(self):
        return len(self.imagepaths)