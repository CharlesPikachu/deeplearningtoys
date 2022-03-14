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
import torch.nn as nn
from torch.utils.data import DataLoader
from modules import Backbone, ImageFolder, LifeGame, Logger, touchdir


'''命令行参数解析'''
def parseArgs():
    parser = argparse.ArgumentParser(description='用卷积神经网络模拟生命游戏')
    parser.add_argument('--batchsize', dest='batchsize', help='批输入大小', type=int, required=False, default=32)
    parser.add_argument('--numworkers', dest='numworkers', help='导入数据时使用的线程数量', type=int, required=False, default=4)
    parser.add_argument('--epochs', dest='epochs', help='迭代数据集的轮次数量', type=int, required=False, default=10)
    parser.add_argument('--logfilepath', dest='logfilepath', help='日志文件保存路径', type=str, required=False, default='train.log')
    parser.add_argument('--ckptdir', dest='ckptdir', help='模型保存的文件夹', type=str, required=False, default='ckpts')
    parser.add_argument('--saveinterval', dest='saveinterval', help='模型保存的epoch间隔', type=int, required=False, default=2)
    parser.add_argument('--lr', dest='lr', help='初始学习率', type=float, required=False, default=0.005)
    args = parser.parse_args()
    return args


'''训练器'''
class Trainer():
    def __init__(self, opts, **kwargs):
        self.opts = opts
    '''模型训练接口'''
    def train(self):
        # 基础准备
        touchdir(self.opts.ckptdir)
        use_cuda = torch.cuda.is_available()
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        logger_handle = Logger(os.path.join(self.opts.ckptdir, self.opts.logfilepath))
        # 定义模型
        model = Backbone()
        if use_cuda: model = model.cuda()
        model.train()
        # 定义数据集
        logger_handle.info('Start to initialize the dataset')
        dataloader = torch.utils.data.DataLoader(
            ImageFolder(),
            batch_size=self.opts.batchsize,
            shuffle=True,
            num_workers=self.opts.numworkers
        )
        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.opts.lr)
        criterion = nn.MSELoss()
        # 模型训练
        # --基础信息打印
        logger_handle.info('Config INFO:')
        for key in vars(self.opts):
            logger_handle.info(f'{key} >>> {getattr(self.opts, key)}')
        # --开始训练
        for epoch in range(1, self.opts.epochs+1):
            for batch_idx, (images, labels) in enumerate(dataloader):
                images, labels = images.type(FloatTensor), labels.type(FloatTensor)
                optimizer.zero_grad()
                preds = model(images)
                loss = criterion(preds, labels)
                logger_handle.info(f'EPOCH: {epoch}/{self.opts.epochs}, BATCH: {batch_idx+1}/{len(dataloader)}, LOSS: {loss.item()}')
                loss.backward()
                optimizer.step()
            if ((epoch % self.opts.saveinterval == 0) and (epoch > 0)) or (epoch == self.opts.epochs):
                ckpt_pth = os.path.join(self.opts.ckptdir, f'epoch_{epoch}.pth')
                torch.save(model.state_dict(), ckpt_pth)


'''run'''
if __name__ == '__main__':
    args = parseArgs()
    client = Trainer(opts=args)
    client.train()