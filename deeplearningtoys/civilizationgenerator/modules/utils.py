'''
Function:
    定义诗歌生成模型
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import torch
import logging
import numpy as np


'''touch dir'''
def touchdir(dirname):
    if not os.path.exists(dirname):
        os.mkdir(dirname)
        return True
    return False


'''创建dataloader'''
def CreateDataloader(tang_resource_path, batch_size=16, num_workers=2):
    poems = np.load(tang_resource_path, allow_pickle=True)
    dataset = poems['data']
    index2word = poems['index2word'].item()
    word2index = poems['word2index'].item()
    dataset = torch.from_numpy(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )
    return dataloader, index2word, word2index


'''日志文件'''
class Logger():
    def __init__(self, logfilepath, **kwargs):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)-8s %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[logging.FileHandler(logfilepath), logging.StreamHandler()])
    @staticmethod
    def log(level, message):
        logging.log(level, message)
    @staticmethod
    def debug(message):
        Logger.log(logging.DEBUG, message)
    @staticmethod
    def info(message):
        Logger.log(logging.INFO, message)
    @staticmethod
    def warning(message):
        Logger.log(logging.WARNING, message)
    @staticmethod
    def error(message):
        Logger.log(logging.ERROR, message)