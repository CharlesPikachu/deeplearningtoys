'''
Function:
    导入数据集
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import glob
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from skimage.transform import resize


'''导入数据集'''
class ImageFolder(Dataset):
    def __init__(self, image_dir, image_size=(350, 350), mode='train', **kwargs):
        assert mode in ['train', 'test']
        self.image_size = image_size
        # 图片
        self.imagepaths = sorted(glob.glob(os.path.join(image_dir, 'Images', '*.*')))
        if mode == 'train': self.imagepaths = self.imagepaths[:int(len(self.imagepaths) * 0.8)]
        else: self.imagepaths = self.imagepaths[int(len(self.imagepaths) * 0.8):]
        if mode == 'train': random.shuffle(self.imagepaths)
        # 标签
        ratings = pd.read_excel(os.path.join(image_dir, 'All_Ratings.xlsx'))
        filenames = ratings.groupby('Filename').size().index.tolist()
        self.labels = []
        for filename in filenames:
            score = round(ratings[ratings['Filename'] == filename]['Rating'].mean(), 2)
            self.labels.append({'Filename': filename, 'score': score})
        self.labels = pd.DataFrame(self.labels)
    '''__getitem__'''
    def __getitem__(self, index):
        # 读取图片
        img_path = self.imagepaths[index]
        img = np.array(Image.open(img_path)) / 255.
        input_img = resize(img, (*self.image_size, 3), mode='reflect')
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = torch.from_numpy(input_img).float()
        # 读取标签
        filename = img_path.split('/')[-1]
        label = self.labels[self.labels.Filename == filename].score.values
        # 返回
        return img_path, input_img, label
    '''__len__'''
    def __len__(self):
        return len(self.imagepaths)