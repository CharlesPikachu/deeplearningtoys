'''
Function:
    train the model
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import torch
import argparse
import torchvision
import torch.nn as nn
from modules import Logging, touchdir, ImageFolder


'''命令行参数解析'''
def parseArgs():
    parser = argparse.ArgumentParser(description='颜值预测模型训练脚本')
    parser.add_argument('--ckptdir', dest='ckptdir', help='模型保存的文件夹', type=str, required=False, default='ckpts')
    parser.add_argument('--imagedir', dest='imagedir', help='数据集路径', type=str, required=False, default='./images/SCUT-FBP5500_v2')
    parser.add_argument('--imagesize', dest='imagesize', help='输入图片大小', type=tuple, required=False, default=(224, 224))
    parser.add_argument('--batchsize', dest='batchsize', help='批输入大小', type=int, required=False, default=32)
    parser.add_argument('--numworkers', dest='numworkers', help='导入数据时使用的线程数量', type=int, required=False, default=4)
    parser.add_argument('--lr', dest='lr', help='初始学习率', type=float, required=False, default=2e-4)
    parser.add_argument('--logfilepath', dest='logfilepath', help='日志文件保存路径', type=str, required=False, default='train.log')
    parser.add_argument('--saveinterval', dest='saveinterval', help='模型保存的epoch间隔', type=int, required=False, default=10)
    parser.add_argument('--epochs', dest='epochs', help='迭代数据集的轮次数量', type=int, required=False, default=80)
    args = parser.parse_args()
    return args


'''训练器'''
class Trainer():
    def __init__(self, opts, **kwargs):
        self.opts = opts
    '''模型训练接口'''
    def train(self):
        # 准备工作
        touchdir(self.opts.ckptdir)
        use_cuda = torch.cuda.is_available()
        # 定义模型
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=1, bias=True)
        if use_cuda: model = model.cuda()
        model.train()
        # 定义数据集
        dataloader = torch.utils.data.DataLoader(
            ImageFolder(image_dir=self.opts.imagedir, image_size=self.opts.imagesize, mode='train'),
            batch_size=self.opts.batchsize,
            shuffle=True,
            num_workers=self.opts.numworkers,
        )
        # 优化器与损失函数
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.opts.lr)
        criterion = nn.MSELoss()
        # 模型训练
        logfilepath = os.path.join(self.opts.ckptdir, self.opts.logfilepath)
        # --基础信息打印
        Logging('Config INFO:', logfilepath)
        for key in vars(self.opts):
            Logging(f'{key} >>> {getattr(self.opts, key)}', logfilepath)
        # --开始训练
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        for epoch in range(1, self.opts.epochs+1):
            Logging(f'Start to train epoch {epoch}/{self.opts.epochs}', logfilepath)
            for batch_idx, (_, imgs, targets) in enumerate(dataloader):
                imgs, targets = imgs.type(FloatTensor), targets.type(FloatTensor)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, targets)
                Logging(f'[{batch_idx+1}/{len(dataloader)}] in Epoch [{epoch}/{self.opts.epochs}]: loss is {loss.item()}', logfilepath)
                loss.backward()
                optimizer.step()
            if ((epoch % self.opts.saveinterval == 0) and (epoch > 0)) or (epoch == self.opts.epochs):
                ckpt_pth = os.path.join(self.opts.ckptdir, f'epoch_{epoch}.pth')
                torch.save(model.state_dict(), ckpt_pth)
                acc = self.test(model, FloatTensor)
                Logging(f'{ckpt_pth} has been saved, the accuracy is {acc}', logfilepath)
            if epoch == int(self.opts.epochs * 3 // 5):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opts.lr * 0.1
            elif epoch == int(self.opts.epochs * 4 // 5):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opts.lr * 0.01
    '''模型测试接口'''
    def test(self, model, FloatTensor):
        model.eval()
        dataloader = torch.utils.data.DataLoader(
            ImageFolder(image_dir=self.opts.imagedir, image_size=self.opts.imagesize, mode='test'),
            batch_size=self.opts.batchsize,
            shuffle=False,
            num_workers=self.opts.numworkers,
        )
        n_correct, n_total = 0, 0
        for batch_idx, (_, imgs, targets) in enumerate(dataloader):
            imgs, targets = imgs.type(FloatTensor), targets.type(FloatTensor)
            preds = model(imgs)
            n_correct += (abs(targets - preds) < 0.5).sum().item()
            n_total += imgs.size(0)
        acc = n_correct / n_total
        model.train()
        return acc


'''run'''
if __name__ == '__main__':
    args = parseArgs()
    client = Trainer(opts=args)
    client.train()