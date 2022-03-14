'''
Function:
    模型测试
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import torch
import imageio
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from modules import Backbone, LifeGame
from matplotlib.animation import FFMpegWriter


'''命令行参数解析'''
def parseArgs():
    parser = argparse.ArgumentParser(description='用卷积神经网络模拟生命游戏')
    parser.add_argument('--frames', dest='frames', help='生成的帧数', type=int, required=False, default=100)
    parser.add_argument('--outputpath', dest='outputpath', help='结果保存路径', type=str, required=False, default='output.gif')
    parser.add_argument('--ckptpath', dest='ckptpath', help='需要导入的模型路径', type=str, required=False, default='cnnlifegame.pth')
    args = parser.parse_args()
    return args


'''模型测试'''
class Inferencer():
    def __init__(self, opts, title='用卷积神经网络模拟生命游戏 —— Charles的皮卡丘', **kwargs):
        self.opts = opts
    '''运行'''
    def run(self):
        # 基础准备
        use_cuda = torch.cuda.is_available()
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        # 定义模型
        model = Backbone()
        model.load_state_dict(torch.load(self.opts.ckptpath, map_location='cpu'))
        if use_cuda: model = model.cuda()
        model.eval()
        # 定义原始生命游戏
        game_client = LifeGame()
        frame_game = game_client.initialize()
        frame_cnn = frame_game.copy()
        input_size, fig, writer = frame_game.shape, plt.figure(), FFMpegWriter()
        with writer.saving(fig, self.opts.outputpath, 100):
            for idx in tqdm(range(1, self.opts.frames)):
                plt.title('Original: Left, CNN: Right, Iteration: %s' % idx)
                ax = fig.add_subplot(121)
                ax.matshow(frame_game, vmin=0, vmax=1, cmap='gray')
                ax = fig.add_subplot(122)
                ax.matshow(frame_cnn, vmin=0, vmax=1, cmap='gray')
                frame_game = game_client.nextframe(frame_game)
                frame_cnn = model(torch.from_numpy(frame_cnn).float().reshape(1, 1, *input_size).type(FloatTensor)).cpu().data.numpy().reshape(input_size)
                writer.grab_frame()
                plt.pause(0.1)
                plt.clf()
                plt.cla()


'''run'''
if __name__ == '__main__':
    args = parseArgs()
    client = Inferencer(opts=args)
    client.run()