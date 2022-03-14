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
from modules import Backbone, loadClasses, saveClasses, Logging, ImageFolder, touchdir


'''命令行参数解析'''
def parseArgs():
    parser = argparse.ArgumentParser(description='垃圾分类模型训练脚本')
    parser.add_argument('--ckptdir', dest='ckptdir', help='模型保存的文件夹', type=str, required=False, default='ckpts')
    parser.add_argument('--imagedir', dest='imagedir', help='数据集路径', type=str, required=False, default='./images/Garbage')
    parser.add_argument('--imagesize', dest='imagesize', help='输入图片大小', type=tuple, required=False, default=(224, 224))
    parser.add_argument('--batchsize', dest='batchsize', help='批输入大小', type=int, required=False, default=32)
    parser.add_argument('--numworkers', dest='numworkers', help='导入数据时使用的线程数量', type=int, required=False, default=4)
    parser.add_argument('--backbone', dest='backbone', help='使用的骨干网络', type=str, required=False, default='resnet50')
    parser.add_argument('--classes', dest='classes', help='垃圾类别数量', type=int, required=False, default=6)
    parser.add_argument('--lr', dest='lr', help='初始学习率', type=float, required=False, default=2e-4)
    parser.add_argument('--logfilepath', dest='logfilepath', help='日志文件保存路径', type=str, required=False, default='train.log')
    parser.add_argument('--saveinterval', dest='saveinterval', help='模型保存的epoch间隔', type=int, required=False, default=10)
    parser.add_argument('--epochs', dest='epochs', help='迭代数据集的轮次数量', type=int, required=False, default=200)
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
        model = Backbone(backbone_type=self.opts.backbone, pretrained=True, num_classes=int(self.opts.classes))
        if use_cuda: model = model.cuda()
        model.train()
        # 定义数据集
        dataset_train = ImageFolder(os.path.join(self.opts.imagedir, 'train'), self.opts.imagesize, True)
        dataloader_train = torch.utils.data.DataLoader(
            dataset_train, 
            batch_size=int(self.opts.batchsize), 
            shuffle=True, 
            num_workers=int(self.opts.numworkers)
        )
        dataset_test = ImageFolder(os.path.join(self.opts.imagedir, 'test'), self.opts.imagesize, False)
        dataloader_test = torch.utils.data.DataLoader(
            dataset_test, 
            batch_size=int(self.opts.batchsize), 
            shuffle=False, 
            num_workers=int(self.opts.numworkers),
        )
        # 优化器与损失函数
        optimizer = torch.optim.Adam(model.parameters(), lr=float(self.opts.lr))
        criterion = nn.CrossEntropyLoss()
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
            for batch_idx, (imgs, labels) in enumerate(dataloader_train):
                imgs, labels = imgs.type(FloatTensor), labels.type(FloatTensor)
                optimizer.zero_grad()
                preds = model(imgs)
                loss = criterion(preds, labels.long())
                Logging(f'[{batch_idx}/{len(dataloader_train)}]: loss is {loss.item()}', logfilepath)
                loss.backward()
                optimizer.step()
            if ((epoch % self.opts.saveinterval == 0) and (epoch > 0)) or (epoch == self.opts.epochs):
                ckpt_pth = os.path.join(self.opts.ckptdir, f'epoch_{epoch}.pth')
                torch.save(model.state_dict(), ckpt_pth)
                acc = self.test(model, dataloader_test, FloatTensor)
                Logging(f'{ckpt_pth} has been saved, the accuracy is {acc}', logfilepath)
            if epoch == int(self.opts.epochs * 3 // 5):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opts.lr * 0.1
            elif epoch == int(self.opts.epochs * 4 // 5):
                for param_group in optimizer.param_groups:
                    param_group['lr'] = self.opts.lr * 0.01
    '''模型测试'''
    def test(self, model, dataloader, FloatTensor):
        model.eval()
        n_correct, n_total = 0, 0
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs, labels = imgs.type(FloatTensor), labels.type(FloatTensor)
            preds = model(imgs)
            n_correct += (preds.max(-1)[1].long() == labels.long()).sum().item()
            n_total += imgs.size(0)
        acc = (n_correct / n_total) * 100
        model.train()
        return acc


'''run'''
if __name__ == '__main__':
    args = parseArgs()
    client = Trainer(opts=args)
    client.train()