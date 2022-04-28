'''
Function:
    训练文明话生成器
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import torch
import argparse
import torch.nn as nn
from modules import Poem, CreateDataloader, Logger, touchdir


'''命令行参数解析'''
def parseArgs():
    parser = argparse.ArgumentParser(description='训练文明话生成器')
    parser.add_argument('--batchsize', dest='batchsize', help='批输入大小', type=int, required=False, default=128)
    parser.add_argument('--numworkers', dest='numworkers', help='导入数据时使用的线程数量', type=int, required=False, default=4)
    parser.add_argument('--epochs', dest='epochs', help='迭代数据集的轮次数量', type=int, required=False, default=100)
    parser.add_argument('--logfilepath', dest='logfilepath', help='日志文件保存路径', type=str, required=False, default='train.log')
    parser.add_argument('--ckptdir', dest='ckptdir', help='模型保存的文件夹', type=str, required=False, default='ckpts')
    parser.add_argument('--saveinterval', dest='saveinterval', help='模型保存的epoch间隔', type=int, required=False, default=1)
    parser.add_argument('--lr', dest='lr', help='初始学习率', type=float, required=False, default=1e-3)
    args = parser.parse_args()
    return args


'''训练文明话生成器'''
class Trainer():
    def __init__(self, cmd_args):
        self.cmd_args = cmd_args
    '''run'''
    def run(self):
        # 初始化
        use_cuda = torch.cuda.is_available()
        rootdir = os.path.split(os.path.abspath(__file__))[0]
        touchdir(self.cmd_args.ckptdir)
        logger_handle = Logger(os.path.join(self.cmd_args.ckptdir, self.cmd_args.logfilepath))
        # 创建数据集
        tang_resource_path = os.path.join(rootdir, 'resources/tang.npz')
        dataloader, index2word, word2index = CreateDataloader(tang_resource_path, self.cmd_args.batchsize, self.cmd_args.numworkers)
        # 创建模型
        model = Poem(len(word2index))
        if use_cuda: model = model.cuda()
        # 创建优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cmd_args.lr)
        # 创建损失函数
        criterion = nn.CrossEntropyLoss()
        # 开始训练
        for epoch in range(1, self.cmd_args.epochs+1):
            for batch_idx, data in enumerate(dataloader):
                optimizer.zero_grad()
                data = data.long().transpose(1, 0).contiguous()
                if use_cuda: data = data.cuda()
                inputs, targets = data[:-1, :], data[1:, :]
                outputs = model(inputs)[0]
                loss = criterion(outputs, targets.view(-1))
                if (batch_idx + 1) % 10 == 0:
                    logger_handle.info(f'[Epoch]: {epoch}/{self.cmd_args.epochs}, [Batch]: {batch_idx+1}/{len(dataloader)}, [Loss]: {loss.item()}')
                loss.backward()
                optimizer.step()
            if epoch % self.cmd_args.saveinterval == 0 or epoch == self.cmd_args.epochs:
                torch.save(model.state_dict(), os.path.join(self.cmd_args.ckptdir, f'epoch_{epoch}.pth'))


'''run'''
if __name__ == '__main__':
    args = parseArgs()
    client = Trainer(args)
    client.run()