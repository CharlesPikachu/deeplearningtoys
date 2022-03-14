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
import logging
import numpy as np


'''新建文件夹'''
def touchdir(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        return False
    return True


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


'''从视频中提取图像'''
def extractImagesFromVideo(videopath, savedir='images', frame_interval=3, target_imgsize=(208, 120), bg_thresh=20):
    touchdir(savedir)
    capture = cv2.VideoCapture(videopath)
    count = 0
    while capture.isOpened():
        ret, img = capture.read()
        if ret:
            count += 1
            if count % frame_interval == 0:
                img = cv2.resize(img, target_imgsize)
                img[np.where(np.greater(img, bg_thresh))] = 255
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                savepath = os.path.join(savedir, str(len(os.listdir(savedir))+1)+'.jpg')
                cv2.imwrite(savepath, img)
        else:
            break


'''保存模型'''
def saveCheckpoints(model, savepath):
    torch.save(model.state_dict(), savepath)
    return True


'''导入模型'''
def loadCheckpoints(model, checkpointspath):
    model.load_state_dict(torch.load(checkpointspath, map_location='cpu'))
    return model