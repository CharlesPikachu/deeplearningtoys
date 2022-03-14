'''
Function:
    数据生成与导入
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import torch
import numpy as np
from tqdm import tqdm


'''模拟生命游戏'''
class LifeGame():
    def __init__(self, max_iters=100, num_games=500, size=20, init_life=100, savedir=None, **kwargs):
        self.max_iters = max_iters
        self.num_games = num_games
        self.size = size
        self.init_life = init_life
        self.savedir = savedir
    '''生成数据'''
    def generate(self):
        images = np.zeros([self.max_iters * self.num_games, 1, self.size, self.size]).astype(np.float32)
        labels = np.zeros([self.max_iters * self.num_games, 1, self.size, self.size]).astype(np.float32)
        for game_idx in tqdm(range(self.num_games)):
            frame = self.initialize(size=self.size, init_life=self.init_life)
            for idx in range(self.max_iters):
                images[idx + game_idx * self.max_iters, 0, :, :] = frame
                frame = self.nextframe(frame)
                labels[idx + game_idx * self.max_iters, 0, :, :] = frame
        if self.savedir is not None:
            np.save(os.path.join(savedir, f'images.npy'), images)
            np.save(os.path.join(savedir, f'labels.npy'), labels)
        return images, labels
    '''初始化'''
    def initialize(self, size=20, init_life=100):
        frame = np.zeros([size, size]).astype(np.float32)
        for i in range(init_life):
            frame[np.random.randint(size), np.random.randint(size)] = 1.
        return frame
    '''下一帧'''
    def nextframe(self, frame):
        next_frame = frame.copy()
        for i in range(0, frame.shape[0]):
            for j in range(0, frame.shape[1]):
                n = frame[max(0, i-1): min(frame.shape[0]-1, i+2), max(0, j-1): min(frame.shape[1]-1, j+2)].sum() - frame[i, j]
                if n == 3:
                    next_frame[i, j] = 1
                elif n == 2:
                    next_frame[i, j] = frame[i, j]
                else:
                    next_frame[i, j] = 0
        return next_frame


'''定义数据集'''
class ImageFolder():
    def __init__(self, **kwargs):
        self.images, self.labels = LifeGame(max_iters=100, num_games=500, size=20, init_life=100).generate()
    '''__getitem__'''
    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]
        image = torch.from_numpy(image).float()
        label = torch.from_numpy(label).float()
        return image, label
    '''__len__'''
    def __len__(self):
        return self.images.shape[0]