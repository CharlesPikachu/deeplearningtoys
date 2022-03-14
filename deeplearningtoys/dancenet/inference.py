'''
Function:
    模型测试
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import os
import cv2
import argparse
import numpy as np
from tqdm import tqdm
from modules import touchdir, Logger, saveCheckpoints, loadCheckpoints, ImageFolder, DanceNet, extractImagesFromVideo


'''命令行参数解析'''
def parseArgs():
    parser = argparse.ArgumentParser(description='神经网络生成会跳舞的小姐姐测试脚本')
    parser.add_argument('--outputpath', dest='outputpath', help='生成的视频保存路径', type=str, required=False, default='output.mp4')
    parser.add_argument('--imagesize', dest='imagesize', help='输入输出的图片大小', type=tuple, required=False, default=(208, 120))
    parser.add_argument('--frames', dest='frames', help='生成的视频总帧数', type=int, required=False, default=1000)
    parser.add_argument('--fps', dest='fps', help='生成的视频FPS', type=int, required=False, default=25)
    parser.add_argument('--logfilepath', dest='logfilepath', help='日志文件保存路径', type=str, required=False, default='test.log')
    parser.add_argument('--ckptpath', dest='ckptpath', help='需要导入的模型路径', type=str, required=False, default='dancenet.pth')
    parser.add_argument('--method', dest='method', help='生成视频的方式, 有random和fromtrain两种', type=str, required=False, default='fromtrain')
    args = parser.parse_args()
    return args


'''模型测试'''
class Inferencer(QWidget):
    def __init__(self, opts, title='颜值预测器 —— Charles的皮卡丘', **kwargs):
        self.opts = opts
    '''运行'''
    def run(self):
        # 基础准备
        logger_handle = Logger(self.opts.logfilepath)
        use_cuda = torch.cuda.is_available()
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        # 模型定义
        model = DanceNet(image_size=self.opts.imagesize)
        model = loadCheckpoints(model, self.opts.ckptpath)
        if use_cuda: model = model.cuda()
        # 开始生成会跳舞的小姐姐
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(self.opts.outputpath, fourcc, self.opts.fps, self.opts.imagesize)
        pbar = tqdm(range(self.opts.frames))
        if self.opts.method == 'random':
            for frame_idx in pbar:
                pbar.set_description('generate image in random')
                z = np.random.normal(0, 1, 128).astype(np.float32)
                z = torch.from_numpy(z).view(1, -1).type(FloatTensor)
                img_gen = model.decoder(z)[0].cpu().data.permute(1, 2, 0).numpy() * 255
                img_gen = img_gen.astype('uint8')
                img_gen = cv2.cvtColor(img_gen, cv2.COLOR_GRAY2RGB)
                video.write(img_gen)
        else:
            for frame_idx in pbar:
                pbar.set_description('generate image in fromtrain')
                img = cv2.imread(os.path.join('images', '%d.jpg' % i), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, self.opts.imagesize)
                img = img.astype(np.float32) / 255
                img = torch.from_numpy(img).unsqueeze(-1).permute(2, 0, 1).unsqueeze(0)
                img_gen = model(img.type(FloatTensor))[0][0].cpu().data.permute(1, 2, 0).numpy() * 255
                img_gen = img_gen.astype('uint8')
                img_gen = cv2.cvtColor(img_gen, cv2.COLOR_GRAY2RGB)
                video.write(img_gen)
        video.release()


'''run'''
if __name__ == '__main__':
    args = parseArgs()
    client = Inferencer(opts=args)
    client.run()