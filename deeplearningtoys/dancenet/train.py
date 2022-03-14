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
from modules import touchdir, Logger, saveCheckpoints, loadCheckpoints, ImageFolder, DanceNet, extractImagesFromVideo


'''命令行参数解析'''
def parseArgs():
    parser = argparse.ArgumentParser(description='神经网络生成会跳舞的小姐姐训练脚本')
    parser.add_argument('--videopath', dest='videopath', help='训练用的视频路径', type=str, required=False, default='videos/dance.mp4')
    parser.add_argument('--logfilepath', dest='logfilepath', help='日志文件保存路径', type=str, required=False, default='train.log')
    parser.add_argument('--ckptdir', dest='ckptdir', help='模型保存的文件夹', type=str, required=False, default='ckpts')
    parser.add_argument('--imagesize', dest='imagesize', help='输入图片大小', type=tuple, required=False, default=(208, 120))
    parser.add_argument('--batchsize', dest='batchsize', help='批输入大小', type=int, required=False, default=64)
    parser.add_argument('--numworkers', dest='numworkers', help='导入数据时使用的线程数量', type=int, required=False, default=4)
    parser.add_argument('--lr', dest='lr', help='初始学习率', type=float, required=False, default=1e-3)
    parser.add_argument('--saveinterval', dest='saveinterval', help='模型保存的epoch间隔', type=int, required=False, default=5)
    parser.add_argument('--epochs', dest='epochs', help='迭代数据集的轮次数量', type=int, required=False, default=50)
    args = parser.parse_args()
    return args


'''训练器'''
class Trainer():
    def __init__(self, opts, **kwargs):
        self.opts = opts
    '''模型训练接口'''
    def train(self):
        # 基础准备
        logfilepath = os.path.join(self.opts.ckptdir, self.opts.logfilepath)
        touchdir(self.opts.ckptdir)
        logger_handle = Logger(logfilepath)
        use_cuda = torch.cuda.is_available()
        # 开始处理视频
        logger_handle.info(f'Start to pre-process the video {self.opts.videopath}')
        extractImagesFromVideo(self.opts.videopath, savedir='images', target_imgsize=self.opts.imagesize)
        # 定义数据集
        dataset = ImageFolder('images', self.opts.imagesize)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.opts.batchsize, shuffle=False, num_workers=self.opts.numworkers)
        # 定义模型
        model = DanceNet(image_size=self.opts.imagesize)
        if use_cuda: model = model.cuda()
        # 定义优化器和损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=self.opts.lr)
        mse_loss_func = nn.MSELoss()
        # 模型训练
        # --基础信息打印
        logger_handle.info('Config INFO:')
        for key in vars(self.opts):
            logger_handle.info(f'{key} >>> {getattr(self.opts, key)}')
        # --开始训练
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        for epoch in range(1, self.opts.epochs+1):
            for batch_idx, img in enumerate(dataloader):
                optimizer.zero_grad()
                img = img.type(FloatTensor)
                img_gen, mean, logvar = model(img)
                mse_loss = mse_loss_func(img_gen, img)
                kl_loss = torch.sum(mean.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)).mul_(-0.5)
                loss = mse_loss + kl_loss
                loss.backward()
                optimizer.step()
                logger_handle.info(f'EPOCH: {epoch}/{self.opts.epochs}, BATCH: {batch_idx+1}/{len(dataloader)}, LOSS: mse_loss {mse_loss.item()}, kl_loss {kl_loss.item()}, loss_total {loss.item()}')
            if ((epoch % self.opts.saveinterval == 0) and (epoch > 0)) or (epoch == self.opts.epochs):
                ckpt_pth = os.path.join(self.opts.ckptdir, f'epoch_{epoch}.pth')
                saveCheckpoints(model, ckpt_pth)
            if epoch == int(self.opts.epochs * 3 // 5):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opts.lr * 0.1
            elif epoch == int(self.opts.epochs * 4 // 5):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opts.lr * 0.01


'''run'''
if __name__ == '__main__':
    args = parseArgs()
    client = Trainer(opts=args)
    client.train()