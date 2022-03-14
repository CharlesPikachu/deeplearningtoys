'''
Function:
    模型测试
Author:
    Charles
微信公众号:
    Charles的皮卡丘
'''
import io
import os
import sys
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtWidgets, QtGui
from modules import Backbone, loadClasses


'''模型测试'''
class Inferencer(QWidget):
    def __init__(self, parent=None, title='垃圾分类器 —— Charles的皮卡丘', **kwargs):
        super(Inferencer, self).__init__(parent)
        # 初始化
        self.rootdir = os.path.split(os.path.abspath(__file__))[0]
        self.setFixedSize(600, 500)
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon(os.path.join(self.rootdir, 'images/icon.png')))
        # 定义一些必要的组件
        grid = QGridLayout()
        # --图片显示用的label
        self.show_label = QLabel()
        self.show_label.setScaledContents(True)
        self.show_label.setMaximumSize(600, 400)
        self.show_image = Image.open(os.path.join(self.rootdir, 'images/assert.jpg')).convert('RGB')
        self.updateimage()
        self.show_image_ext = 'jpg'
        # --图片选择
        self.image_btn = QPushButton('选择图片')
        self.image_label = QLabel('图片路径:')
        self.image_edit = QLineEdit()
        self.image_edit.setText(os.path.join(self.rootdir, 'images/assert.jpg'))
        # --图片识别
        self.classify_btn = QPushButton('垃圾识别')
        self.classify_label = QLabel('识别结果:')
        self.classify_edit = QLineEdit()
        self.classify_edit.setText('暂无识别结果')
        # 组件布局
        grid.addWidget(self.show_label, 0, 0, 5, 7)
        grid.addWidget(self.image_label, 5, 0, 1, 1)
        grid.addWidget(self.image_edit, 5, 1, 1, 5)
        grid.addWidget(self.image_btn, 5, 6, 1, 1)
        grid.addWidget(self.classify_label, 6, 0, 1, 1)
        grid.addWidget(self.classify_edit, 6, 1, 1, 5)
        grid.addWidget(self.classify_btn, 6, 6, 1, 1)
        self.setLayout(grid)
        # 事件绑定
        self.classify_btn.clicked.connect(self.classify)
        self.image_btn.clicked.connect(self.openimage)
    '''更新界面上的图片'''
    def updateimage(self):
        if self.show_image is None: return
        fp = io.BytesIO()
        self.show_image.save(fp, 'JPEG')
        qtimage = QtGui.QImage()
        qtimage.loadFromData(fp.getvalue(), 'JPEG')
        qtimage_pixmap = QtGui.QPixmap.fromImage(qtimage)
        self.show_label.setPixmap(qtimage_pixmap)
    '''打开图片'''
    def openimage(self):
        filepath = QFileDialog.getOpenFileName(self, '请选取图片路径', self.rootdir)
        self.image_edit.setText(filepath[0])
    '''分类'''
    def classify(self):
        # 初始化
        use_cuda = torch.cuda.is_available()
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        classnames = loadClasses(os.path.join(self.rootdir, 'images/classes.data'))
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # 导入模型
        model = Backbone(backbone_type='resnet50', pretrained=False, num_classes=6)
        model.load_state_dict(model_zoo.load_url('https://github.com/CharlesPikachu/deeplearningtoys/releases/download/garbageclassifier/garbageclassifier.pth', map_location='cpu'))
        if use_cuda: model = model.cuda()
        model.eval()
        # 开始识别
        img = Image.open(self.image_edit.text())
        img_input = transform(img)
        img_input = img_input.type(FloatTensor).unsqueeze(0)
        with torch.no_grad(): preds = model(img_input)
        preds = nn.Softmax(-1)(preds).cpu()
        max_prob, max_prob_id = preds.view(-1).max(0)
        max_prob = max_prob.item()
        max_prob_id = max_prob_id.item()
        clsname = classnames[max_prob_id]
        self.classify_edit.setText(f'识别结果: {clsname}, 置信度: {max_prob}')


'''run'''
if __name__ == '__main__':
    app = QApplication(sys.argv)
    client = Inferencer()
    client.show()
    sys.exit(app.exec_())